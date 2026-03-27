from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any, Dict, List

from smolagents.monitoring import LogLevel

try:
    from ..citation_validator import add_citations
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from citation_validator import add_citations

top_number = 10
class KeywordSearchPlanningService:
    
    """关键词检索与大纲输入构建服务。

    说明：
    - 仅承载“关键词检索/扩展检索/大纲输入组装”实现；
    - 由 LongWriterAgent 负责调度、状态管理与日志归口；
    - 组件间通信仍通过 JSON 字符串在上层 orchestrator 中完成。
    """
    @staticmethod
    def enrich_missing_keywords_for_items(agent, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        主要作用：为缺失关键词的论文条目补齐关键词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - papers (List[Dict[str, Any]]): 论文材料列表。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        if not papers:
            return []
        try:
            response = agent.execute_tool_call("keyword_extraction", {"input": json.dumps(papers, ensure_ascii=False)})
            parsed = agent._parse_json(str(response))
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            agent.logger.log(f"⚠️ 关键词总结失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def summarize_five_keywords_from_non_papers(
        agent,
        papers: List[Dict[str, Any]],
        initial_concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        主要作用：从非论文材料中补充关键词线索。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - papers (List[Dict[str, Any]]): 论文材料列表。
        - initial_concepts (List[Dict[str, Any]]): 任务初步抽取得到的概念集合。

        返回值：
        - List[str]：返回列表结果，通常表示章节、论文、概念或日志记录集合。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        if not papers:
            return []
        try:
            response = agent.execute_tool_call("keyword_extraction", {"input": json.dumps(papers, ensure_ascii=False)})
            parsed = agent._parse_json(str(response))
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            agent.logger.log(f"⚠️ 非论文关键词总结失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def extract_initial_concepts(agent, task: str) -> List[Dict[str, Any]]:
        """
        主要作用：从任务中抽取第一轮检索概念。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - task (str): 用户给出的原始写作任务，或经清洗后的任务描述。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """

        try:
            agent.logger.log("📍 [第1步] 初始概念拆解...", level=LogLevel.INFO)
            skill_input = f"用户研究主题：{task}"
            response = agent.execute_tool_call("initial_concept_decompose", {"input": skill_input})
            concepts = agent._parse_json(str(response))

            if not isinstance(concepts, list) or len(concepts) == 0:
                raise ValueError("概念拆解结果无效：非列表或为空")

            agent.logger.log(f"✅ 成功拆解为 {len(concepts)} 个概念组", level=LogLevel.INFO)
            for i, concept in enumerate(concepts, 1):
                name = concept.get("concept_name", "")
                kws = concept.get("keywords", [])
                agent.logger.log(f"  {i}. {name}: {', '.join(kws[:3])}...", level=LogLevel.INFO)
            return concepts
        except Exception as e:
            agent.logger.log(f"⚠️ 概念拆解失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def first_shot_retrieval(agent, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        主要作用：执行第一轮检索，获取高价值论文集合。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - concepts (List[Dict[str, Any]]): 概念集合，通常包含 concept_name 与 keywords。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

        try:
            agent.logger.log("🔍 [第2步] First-Shot 初探检索...", level=LogLevel.INFO)

            # 【修改后：DuckDuckGo 专属布尔逻辑】
            query_parts = []
            for concept in concepts:
                keywords = concept.get("keywords", [])
                if keywords:
                    # 强制必须包含每个概念组的第一个/最核心的词 (使用 +)
                    core_kw = f'+"{keywords[0]}"'
                    # 同组的其他词作为补充上下文（空格连接）
                    other_kws = " ".join([f'"{kw}"' for kw in keywords[1:3]]) 
                    query_parts.append(f"{core_kw} {other_kws}".strip())
                    
            # 直接用空格拼接各个概念组
            search_query = " ".join(query_parts)
            # 生成效果类似：+"人工智能" "AI" +"医疗" "诊断"
            agent.logger.log(f"📝 检索式: {search_query}", level=LogLevel.DEBUG)

            if agent.coarse_rag_mode == "web_agent":
                tagged_result = agent._tagged_search_service.run_tagged_web_agent_search(
                    agent,
                    query=search_query,
                    top_k=top_number,
                )
                # 获取 available_citations 字典
                citations = tagged_result.get("available_citations", {})

                # 遍历所有论文
                papers = []
                for title, paper_info in citations.items():
                    papers.append({title: paper_info})

                # 限制返回数量
                papers = papers[:top_number]
                agent.state[agent.STATE_COARSE_RAG_CONTEXT] = tagged_result.get("available_citations", "")
                agent.logger.log(f"✅ tagged web_agent 检索到 {len(papers)} 篇候选资料", level=LogLevel.INFO)
                agent.logger.log(papers)
                return papers[:top_number]

            if agent.coarse_rag_mode == "web_search":
                result = agent.execute_tool_call(agent.search_tool_name, {"query": search_query})
                papers = KeywordSearchPlanningService.parse_search_results_to_papers(str(result))
                papers = KeywordSearchPlanningService.enrich_missing_keywords_for_items(agent, papers)
                agent.logger.log(f"✅ 检索到 {len(papers)} 篇文献", level=LogLevel.INFO)
                return papers[:top_number]

            raise ValueError(f"不支持的 coarse_rag_mode: {agent.coarse_rag_mode}")
        except Exception as e:
            agent.logger.log(f"⚠️ First-Shot 检索失败: {e}", level=LogLevel.ERROR)
            raise


    @staticmethod
    def extract_and_filter_keywords(agent, papers: List[Dict[str, Any]], initial_concepts: List[Dict[str, Any]]) -> List[str]:
        """
        主要作用：从检索材料中提取并筛选扩展关键词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - papers (List[Dict[str, Any]]): 论文材料列表。
        - initial_concepts (List[Dict[str, Any]]): 任务初步抽取得到的概念集合。

        返回值：
        - List[str]：返回列表结果，通常表示章节、论文、概念或日志记录集合。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """

        try:
            agent.logger.log("📊 [第3步] 术语提取与频次统计...", level=LogLevel.INFO)
            if not papers:
                raise ValueError("术语提取失败：无文献数据")

            print(f"papers: {papers}")
            papers_payload = json.dumps(papers, ensure_ascii=False)

            response = agent.execute_tool_call("keyword_extraction", {"input": papers_payload})
            response_json = agent._parse_json(str(response))
            existing_keywords = set()

            for concept in initial_concepts:
                # 提取关键词列表，过滤空值，添加到集合
                keywords = [str(kw).strip() for kw in concept.get("keywords", []) if str(kw).strip()]
                existing_keywords.update(keywords)
            agent.logger.log(f"🔍 已有关键词（{len(existing_keywords)} 个）: {existing_keywords}", level=LogLevel.INFO)

            all_keywords = []
            for paper_dict in response_json:  # response 是列表，每个元素是字典
                for paper_title, paper_info in paper_dict.items():  # 遍历每个字典的键值对
                    keywords = paper_info.get("key_words", [])
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
            agent.logger.log(f"🔍 从检索结果中提取到 {len(all_keywords)} 个关键词: {all_keywords}", level=LogLevel.INFO)

            # 统计频次
            term_counter = Counter(all_keywords)

            # 提取频次 >=2 且不在 existing_keywords 中的关键词
            candidate_words = [term for term, count in term_counter.items() 
                            if count >= 2 and term not in existing_keywords]

            candidate_words = [term for term, count in term_counter.items() if count >= 2 and term not in existing_keywords]
            agent.logger.log(f"🔍 提取到 {len(candidate_words)} 个候选关键词: {candidate_words}", level=LogLevel.INFO)
            return candidate_words
        except Exception as e:
            agent.logger.log(f"⚠️ 术语提取失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def upgrade_concepts_with_keywords(
        agent,
        initial_concepts: List[Dict[str, Any]],
        candidate_words: List[str],
    ) -> List[Dict[str, Any]]:
        """
        主要作用：将新关键词并入初始概念，形成升级概念。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - initial_concepts (List[Dict[str, Any]]): 任务初步抽取得到的概念集合。
        - candidate_words (List[str]): 准备并入概念集合的候选关键词。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        try:
            agent.logger.log("🔗 [第4步] 语义甄别与概念升级...", level=LogLevel.INFO)
            if not candidate_words:
                agent.logger.log("⚠️ 无候选关键词可用，跳过概念升级", level=LogLevel.INFO)
                return initial_concepts
            agent.logger.log(f"initial_concepts:{initial_concepts}")
            agent.logger.log(f"candidate_words:{candidate_words}")
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
                + "\n\n我们在最新文献中发现以下高频候选词：\n"
                + candidate_hint
            )

            response = agent.execute_tool_call("concept_decompose", {"input": skill_input})
            upgraded_concepts = agent._parse_json(str(response))

            if not isinstance(upgraded_concepts, list) or len(upgraded_concepts) == 0:
                raise ValueError("概念升级失败：结果无效")
            return upgraded_concepts
        except Exception as e:
            agent.logger.log(f"⚠️ 概念升级失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def second_shot_retrieval(agent, upgraded_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        主要作用：执行第二轮检索，补强材料覆盖面。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - upgraded_concepts (List[Dict[str, Any]]): 融合扩展关键词后的升级概念集合。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

        try:
            agent.logger.log("🎯 [第5步] Second-Shot 精准检索...", level=LogLevel.INFO)
            # 【修改后：DuckDuckGo 专属布尔逻辑】
            query_parts = []
            for concept in upgraded_concepts:
                keywords = concept.get("keywords", [])
                if keywords:
                    # 强制必须包含每个概念组的第一个/最核心的词 (使用 +)
                    core_kw = f'+"{keywords[0]}"'
                    # 同组的其他词作为补充上下文（空格连接）
                    other_kws = " ".join([f'"{kw}"' for kw in keywords[1:3]]) 
                    query_parts.append(f"{core_kw} {other_kws}".strip())
                    
            # 直接用空格拼接各个概念组
            search_query = " ".join(query_parts)
            # 生成效果类似：+"人工智能" "AI" +"医疗" "诊断"
            if agent.coarse_rag_mode == "web_agent":
                tagged_result = agent._tagged_search_service.run_tagged_web_agent_search(
                    agent,
                    query=search_query,
                    top_k=top_number,
                )
                # 获取 available_citations 字典
                citations = tagged_result.get("available_citations", {})

                # 遍历所有论文
                papers = []
                for title, paper_info in citations.items():
                    papers.append({title: paper_info})

                plan_citations = citations
                if plan_citations:
                    add_citations(agent, plan_citations, "规划阶段/Second-Shot")

                papers = KeywordSearchPlanningService.enrich_missing_keywords_for_items(agent, papers)
                agent.state[agent.STATE_COARSE_RAG_CONTEXT] = tagged_result.get("available_citations", "")
                agent.logger.log(f"✅ tagged web_agent Second-Shot 检索到 {len(papers)} 篇候选资料", level=LogLevel.INFO)
                return papers

            if agent.coarse_rag_mode == "web_search":
                result = agent.execute_tool_call(agent.search_tool_name, {"query": search_query})
                papers = KeywordSearchPlanningService.parse_search_results_to_papers(str(result))
                papers = KeywordSearchPlanningService.enrich_missing_keywords_for_items(agent, papers)
                agent.logger.log(f"✅ Second-Shot 检索到 {len(papers)} 篇文献", level=LogLevel.INFO)
                return papers

            raise ValueError(f"不支持的 coarse_rag_mode: {agent.coarse_rag_mode}")
        except Exception as e:
            agent.logger.log(f"⚠️ Second-Shot 检索失败: {e}", level=LogLevel.ERROR)
            raise



def main() -> int:
    """
    主要作用：执行 main 相关逻辑。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """

    parser = argparse.ArgumentParser(description="KeywordSearchPlanningService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("KeywordSearchPlanningService static methods:")
        print("- extract_initial_concepts(agent, task)")
        print("- first_shot_retrieval(agent, concepts)")
        print("- parse_search_results_to_papers(text)")
        print("- extract_and_filter_keywords(agent, papers, initial_concepts)")
        print("- upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words)")
        print("- second_shot_retrieval(agent, upgraded_concepts)")
        print("- format_paper_materials(agent, papers, limit=12)")
        return 0

    print("这是 service 文件，不是可直接执行工作流组件。")
    print("请运行组件文件进行命令行调试，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_keyword_search.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
