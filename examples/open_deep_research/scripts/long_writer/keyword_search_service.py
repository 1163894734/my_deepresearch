from __future__ import annotations

import argparse
from collections import Counter
import json
import urllib.parse
import datetime
import time
import re
from typing import Any, Dict, List, TYPE_CHECKING
import concurrent.futures

from smolagents.monitoring import LogLevel
from smolagents.models import ChatMessage, MessageRole
# 🔥 彻底去除了 execute_tool_call 的导入，只保留安全的 JSON 解析
from utils.common_utils import safe_json_parse

if TYPE_CHECKING:
    from scripts.multi_agent.agent_context import PipelineContext

try:
    from ..citation_validator import add_citations
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from citation_validator import add_citations

top_number = 10

class KeywordSearchPlanningService:
    """基于双引擎 (ArXiv / OpenAlex) 学术检索的五步关键词扩展大纲构建服务"""

    @staticmethod
    def _extract_time_constraint(context: "PipelineContext", task: str) -> tuple[int, int]:
        """利用大模型从原始任务中提取显性或隐性的年份限制"""
        current_year = datetime.datetime.now().year
        
        prompt = f"""
        请分析用户的检索任务，提取严格的年份限制。
        任务: "{task}"
        当前年份: {current_year}
        
        提取规则：
        1. 显性限制：如果用户明确说了“最近N年”、“2022年之后”，请推算出精确的 year_start 和 year_end。
        2. 隐性限制（⚠️极其重要）：如果用户使用了诸如“前沿”、“最新”、“近期”、“进展”、“现状”、“近期进展”等暗示需要最新文献的词汇，请自动将时间限制设定为【最近3年】（即 year_start: {current_year - 2}, year_end: {current_year}）。
        3. 无限制：如果完全没有时间暗示，请输出 year_start: 0, year_end: 9999。
        
        必须仅输出合法 JSON，例如：
        {{"year_start": 2024, "year_end": 2026}}
        """
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            response = context.model(messages).content
            parsed = safe_json_parse(str(response))
            return int(parsed.get("year_start", 0)), int(parsed.get("year_end", 9999))
        except Exception as e:
            context.logger.log(f"⚠️ 年份提取失败，默认不限制时间: {e}", level=LogLevel.INFO)
            return 0, 9999

    @staticmethod
    def _build_arxiv_query(concepts: List[Dict[str, Any]]) -> str:
        """根据概念组确定性地生成带有完美括号的 ArXiv 布尔检索式"""
        query_parts = []
        for concept in concepts:
            keywords = concept.get("keywords", [])
            if not keywords:
                continue
            # 取每组的前3个核心词，包装成 all:"xxx" 的形式
            or_parts = [f'all:"{str(kw).strip()}"' for kw in keywords[:3]]
            # 组内用 OR 连接，并加上严格的括号
            concept_query = "(" + " OR ".join(or_parts) + ")"
            query_parts.append(concept_query)
        
        # 组间用 AND 强制交集
        final_query = " AND ".join(query_parts)
        return final_query

    # ================= 以下为五步核心工作流 =================

    @staticmethod
    def extract_initial_concepts(context: "PipelineContext", task: str) -> List[Dict[str, Any]]:
        """第1步：从任务中抽取第一轮检索概念，直接调用大模型确保按领域+技术方向拆分"""
        try:
            engine_name = context.state.get('search_engine', 'arxiv').upper()
            context.logger.log(f"📍 [第1步] 初始概念拆解与时间约束分析 ({engine_name} 模式)...", level=LogLevel.INFO)
            
            # 提取时间限制并存入状态
            y_start, y_end = KeywordSearchPlanningService._extract_time_constraint(context, task)
            context.state["arxiv_year_start"] = y_start
            context.state["arxiv_year_end"] = y_end
            if y_start > 0:
                context.logger.log(f"⏱️ 提取到时间限制: {y_start} - {y_end}年", level=LogLevel.INFO)

            prompt = f"""
            请分析用户的研究主题，提取用于 {engine_name} 学术数据库的精确检索概念。
            用户研究主题："{task}"
            
            【极其重要的拆解与过滤规则 - 必读！】：
            1. 领域基座 (Domain)：提取最宏观的领域名词，例如 "large language model", "computer vision"。
            2. 具体技术方向 (Technical Direction)：提取该任务最核心的具体技术名词。
               🚨【致命错误警告】：绝对禁止将“前沿”、“进展”、“最新”、“现状”、“趋势”、“cutting-edge”、“state-of-the-art”、“advancement”等【修饰性虚词/元词】作为检索词！真实的学术论文绝少使用这些营销词汇！
               
               ✅【正确的拯救做法】：
               - 如果用户明确指出了具体技术（如“大模型推理”），则正常提取 "reasoning"。
               - 如果用户只是泛泛地问“大模型前沿进展/趋势”，请你凭借你的学术知识，【直接替用户写出】当前该领域最火的 2 到 3 个【真实硬核技术词汇】作为代替（例如对大模型来说，你可以写 "RLHF", "agent", "reasoning", "MoE"）。
               - 如果这是一篇宏观调研，可以加上 "survey" 或 "review"。
            
            要求：
            - 必须将中文意图翻译为最准确、标准的【英文学术关键词】。
            - 必须输出纯 JSON 数组格式，绝对不要输出 Markdown 代码块或其他多余解释。
            
            示例输出：
            [
                {{"concept_name": "领域基座", "keywords": ["large language model", "LLM"]}},
                {{"concept_name": "具体技术方向", "keywords": ["agent", "reasoning", "alignment"]}}
            ]
            """
            
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            response = context.model(messages).content
            concepts = safe_json_parse(str(response))

            if not isinstance(concepts, list) or len(concepts) == 0:
                raise ValueError("概念拆解结果无效：非列表或为空")

            context.logger.log(f"✅ 成功拆解为 {len(concepts)} 个概念组", level=LogLevel.INFO)
            for i, concept in enumerate(concepts, 1):
                name = concept.get("concept_name", "")
                kws = concept.get("keywords", [])
                context.logger.log(f"  {i}. {name}: {', '.join(kws[:3])}...", level=LogLevel.INFO)
            return concepts
        except Exception as e:
            context.logger.log(f"⚠️ 概念拆解失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def first_shot_retrieval(context: "PipelineContext", concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第2步：利用初始概念执行 First-Shot 检索，接入双引擎底座"""
        try:
            engine = context.state.get('search_engine', 'arxiv').lower()
            context.logger.log(f"🔍 [第2步] First-Shot {engine.upper()} 初探检索...", level=LogLevel.INFO)

            if engine == "openalex":
                kws = []
                for c in concepts:
                    if c.get("keywords"):
                        kw = str(c["keywords"][0]).strip()
                        if " " in kw and not kw.startswith('"'):
                            kw = f'"{kw}"'
                        kws.append(kw)
                search_query = " ".join(kws)
            else:
                search_query = KeywordSearchPlanningService._build_arxiv_query(concepts)

            context.logger.log(f"📝 生成精确检索式: {search_query}", level=LogLevel.DEBUG)

            y_start = context.state.get("arxiv_year_start", 0)
            y_end = context.state.get("arxiv_year_end", 9999)

            try:
                from .academic_search_service import AcademicSearchService
            except ImportError:
                from examples.open_deep_research.scripts.long_writer.academic_search_service import AcademicSearchService

            citations_dict = AcademicSearchService.search_academic_papers(
                context=context, # 🔥 已修改为 context
                search_query=search_query,
                year_start=str(y_start) if y_start > 0 else "",
                year_end=str(y_end) if y_end > 0 else "",
                max_fetch=15, 
                target_count=top_number
            )

            papers = [{title: info} for title, info in citations_dict.items()]

            context.logger.log(f"✅ First-Shot 从 {engine.upper()} 检索到 {len(papers)} 篇真实文献", level=LogLevel.INFO)
            return papers
        except Exception as e:
            context.logger.log(f"⚠️ First-Shot 检索失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def _extract_single_paper_keywords(context: "PipelineContext", title: str, abstract: str) -> List[str]:
        """直接调用底层大模型进行单篇文献的关键词提取"""
        prompt = f"""
        你是一位严谨的学术信息抽取专家。请从以下论文标题和摘要中提取 3-5 个核心学术关键词（统一为标准缩写，例如LLM）。
        【标题】: {title}
        【摘要】: {abstract}
        
        ⚠️ 必须严格输出纯 JSON 字符串数组，不要任何解释！格式：["词1", "词2"]
        """
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            response = context.model(messages, temperature=0.1).content
            parsed = safe_json_parse(str(response))
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            context.logger.log(f"⚠️ 单篇提取失败: {e}", level=LogLevel.INFO)
            return []

    @staticmethod
    def extract_and_filter_keywords(context: "PipelineContext", papers: List[Dict[str, Any]], initial_concepts: List[Dict[str, Any]]) -> List[str]:
        """第3步：并行从摘要中提取学术关键词"""
        try:
            context.logger.log("📊 [第3步] 术语提取与频次统计 (并行提取中)...", level=LogLevel.INFO)
            if not papers:
                return []

            tasks = []
            for paper_dict in papers:
                for title, info in paper_dict.items():
                    abstract = info.get("abstract", "")
                    if abstract:
                        tasks.append((title, abstract))

            all_keywords = []
            
            max_workers = min(10, len(tasks)) if tasks else 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_title = {
                    executor.submit(KeywordSearchPlanningService._extract_single_paper_keywords, context, t, a): t 
                    for t, a in tasks
                }
                
                for future in concurrent.futures.as_completed(future_to_title):
                    title = future_to_title[future]
                    kws = future.result()
                    all_keywords.extend(kws)
                    
                    for paper_dict in papers:
                        if title in paper_dict:
                            paper_dict[title]["key_words"] = kws

            existing_keywords = {str(kw).strip() for concept in initial_concepts for kw in concept.get("keywords", []) if str(kw).strip()}
            
            term_counter = Counter(all_keywords)
            candidate_words = [term for term, count in term_counter.items() 
                            if count >= 2 and term not in existing_keywords]

            context.logger.log(f"🔍 挖掘出 {len(candidate_words)} 个高频新候选词: {candidate_words}", level=LogLevel.INFO)
            return candidate_words
            
        except Exception as e:
            context.logger.log(f"⚠️ 术语提取失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def upgrade_concepts_with_keywords(
        context: "PipelineContext",
        initial_concepts: List[Dict[str, Any]],
        candidate_words: List[str],
    ) -> List[Dict[str, Any]]:
        """第4步：将新术语融合进初始概念"""
        try:
            context.logger.log("🔗 [第4步] 语义甄别与概念升级...", level=LogLevel.INFO)
            if not candidate_words:
                context.logger.log("⚠️ 无候选关键词可用，跳过概念升级", level=LogLevel.INFO)
                return initial_concepts

            concepts_lines = []
            for idx, concept in enumerate(initial_concepts, 1):
                concept_name = concept.get("concept_name", f"概念组{idx}")
                keywords = concept.get("keywords", [])
                group_label = chr(ord("A") + idx - 1)
                concepts_lines.append(f"组{group_label}（{concept_name}）：{json.dumps(keywords, ensure_ascii=False)}")

            candidate_hint = json.dumps(candidate_words, ensure_ascii=False)
            skill_input = (
                "目前我们有以下核心概念组：\n"
                + "\n".join(concepts_lines)
                + "\n\n我们在最新文献摘要中发现以下高频候选词：\n"
                + candidate_hint
            )

            # 🔥 极简的网关调用，消灭了冗余参数
            response = context.execute_tool_call("concept_decompose", {"input": skill_input})
            upgraded_concepts = safe_json_parse(str(response))

            if not isinstance(upgraded_concepts, list) or len(upgraded_concepts) == 0:
                raise ValueError("概念升级失败：结果无效")
            
            context.logger.log(f"✅ 概念升级完毕", level=LogLevel.INFO)
            return upgraded_concepts
        except Exception as e:
            context.logger.log(f"⚠️ 概念升级失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def second_shot_retrieval(context: "PipelineContext", upgraded_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第5步：使用升级后的概念组执行精确检索，接入双引擎底座"""
        try:
            engine = context.state.get('search_engine', 'arxiv').lower()
            context.logger.log(f"🎯 [第5步] Second-Shot {engine.upper()} 精准检索...", level=LogLevel.INFO)
            
            if engine == "openalex":
                kws = []
                for c in upgraded_concepts:
                    if c.get("keywords"):
                        kw = str(c["keywords"][0]).strip()
                        if " " in kw and not kw.startswith('"'):
                            kw = f'"{kw}"'
                        kws.append(kw)
                search_query = " ".join(kws)
            else:
                search_query = KeywordSearchPlanningService._build_arxiv_query(upgraded_concepts)

            context.logger.log(f"📝 升级后的检索式: {search_query}", level=LogLevel.DEBUG)

            y_start = context.state.get("arxiv_year_start", 0)
            y_end = context.state.get("arxiv_year_end", 9999)

            try:
                from .academic_search_service import AcademicSearchService
            except ImportError:
                from examples.open_deep_research.scripts.long_writer.academic_search_service import AcademicSearchService

            citations_dict = AcademicSearchService.search_academic_papers(
                context=context, # 🔥 已修改为 context
                search_query=search_query,
                year_start=str(y_start) if y_start > 0 else "",
                year_end=str(y_end) if y_end > 0 else "",
                max_fetch=20, 
                target_count=top_number
            )

            papers = [{title: info} for title, info in citations_dict.items()]

            context.logger.log(f"✅ Second-Shot 从 {engine.upper()} 精准检索到 {len(papers)} 篇最终文献", level=LogLevel.INFO)
            
            if papers:
                flat_citations = {k: v for d in papers for k, v in d.items()}
                # 这一步依赖 citation_validator.add_citations，只要那里的签名也适配了就行
                add_citations(context, flat_citations, "规划阶段/Second-Shot")
                
            return papers
        except Exception as e:
            context.logger.log(f"⚠️ Second-Shot 检索失败: {e}", level=LogLevel.ERROR)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="KeywordSearchPlanningService 双引擎版调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("KeywordSearchPlanningService static methods:")
        print("- extract_initial_concepts(context, task)")
        print("- first_shot_retrieval(context, concepts)")
        print("- extract_and_filter_keywords(context, papers, initial_concepts)")
        print("- upgrade_concepts_with_keywords(context, initial_concepts, candidate_words)")
        print("- second_shot_retrieval(context, upgraded_concepts)")
        return 0

    print("这是 service 文件，请运行组件文件进行测试。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())