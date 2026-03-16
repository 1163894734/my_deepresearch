from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from smolagents.monitoring import LogLevel


class KeywordSearchPlanningService:
    """关键词检索与大纲输入构建服务。

    说明：
    - 仅承载“关键词检索/扩展检索/大纲输入组装”实现；
    - 由 LongWriterAgent 负责调度、状态管理与日志归口；
    - 组件间通信仍通过 JSON 字符串在上层 orchestrator 中完成。
    """

    @staticmethod
    def _normalize_paper_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """将 tagged/web 搜索条目统一成规划阶段的文献素材结构。"""
        title = str(item.get("title", "")).strip()
        abstract_parts = [
            str(item.get("content_summary", "")).strip(),
            str(item.get("evidence_quote", "")).strip(),
        ]
        abstract = " ".join([p for p in abstract_parts if p]).strip()

        return {
            "title": title[:200] if title else "Untitled source",
            "abstract": abstract[:1000] if abstract else "（无摘要）",
            "keywords": item.get("keywords", []) if isinstance(item.get("keywords", []), list) else [],
            "authors": str(item.get("authors", "")).strip() or "Unknown",
            "url": str(item.get("url", "")).strip() or "N/A",
            "published_time": str(item.get("time", "")).strip() or "unknown",
            "year": str(item.get("year", "")).strip(),
            "source_type": str(item.get("source_type", "webpage")).strip() or "webpage",
            "apa_citation": str(item.get("apa_citation", "")).strip(),
        }

    @staticmethod
    def _is_paper_like(item: Dict[str, Any]) -> bool:
        source_type = str(item.get("source_type", "")).strip().lower()
        if source_type in {"paper", "preprint", "journal", "arxiv", "conference", "thesis"}:
            return True

        has_author = bool(str(item.get("authors", "")).strip())
        has_year = bool(str(item.get("year", "")).strip())
        has_apa = bool(str(item.get("apa_citation", "")).strip())
        return (has_author and has_year) or has_apa

    @staticmethod
    def _summarize_five_keywords_for_item(agent, item: Dict[str, Any]) -> List[str]:
        """当单条检索结果缺少关键词时，调用模型补充 5 个关键词。"""
        title = str(item.get("title", "")).strip()
        abstract = str(item.get("abstract", "")).strip()
        source_type = str(item.get("source_type", "webpage")).strip() or "webpage"

        prompt = (
            "请基于以下单条检索结果，总结恰好5个关键词。\\n"
            "要求：\\n"
            "1) 只返回 JSON 数组，例如 [\"关键词1\", \"关键词2\", ...]；\\n"
            "2) 必须返回恰好5个关键词；\\n"
            "3) 关键词应可用于后续检索，避免过泛词。\\n\\n"
            f"SourceType: {source_type}\\n"
            f"Title: {title or 'Untitled'}\\n"
            f"Summary: {abstract or '（无摘要）'}"
        )

        try:
            response = agent.execute_tool_call("keyword_extraction", {"input": prompt})
            parsed = agent._parse_json(str(response))
            if not isinstance(parsed, list):
                return []

            cleaned: List[str] = []
            seen = set()
            for kw in parsed:
                kw_text = str(kw).strip()
                if not kw_text or kw_text in seen:
                    continue
                seen.add(kw_text)
                cleaned.append(kw_text)
                if len(cleaned) >= 5:
                    break
            return cleaned
        except Exception as e:
            agent.logger.log(f"⚠️ 单条关键词补全失败: {e}", level=LogLevel.INFO)
            return []

    @staticmethod
    def enrich_missing_keywords_for_items(agent, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对缺少关键词的检索条目逐条补全关键词。"""
        enriched: List[Dict[str, Any]] = []
        filled = 0
        for paper in papers:
            if not isinstance(paper, dict):
                continue

            item = dict(paper)
            keywords = item.get("keywords", [])
            keyword_list = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []

            if not keyword_list:
                backfilled = KeywordSearchPlanningService._summarize_five_keywords_for_item(agent, item)
                if backfilled:
                    item["keywords"] = backfilled
                    filled += 1
                else:
                    item["keywords"] = []
            else:
                item["keywords"] = keyword_list[:5]

            enriched.append(item)

        if filled:
            agent.logger.log(f"ℹ️ 已为 {filled} 条缺少关键词的检索结果补全5关键词", level=LogLevel.INFO)
        return enriched

    @staticmethod
    def summarize_five_keywords_from_non_papers(
        agent,
        papers: List[Dict[str, Any]],
        initial_concepts: List[Dict[str, Any]],
    ) -> List[str]:
        """当检索结果非论文为主时，让模型直接总结 5 个关键词。"""
        if not papers:
            return []

        snippets: List[str] = []
        for idx, paper in enumerate(papers[:20], 1):
            title = str(paper.get("title", "")).strip()
            summary = str(paper.get("abstract", "")).strip()
            source_type = str(paper.get("source_type", "webpage")).strip() or "webpage"
            snippets.append(f"[{idx}] ({source_type}) {title} | {summary}")

        existing_keywords = set()
        for concept in initial_concepts:
            existing_keywords.update(str(k).strip() for k in concept.get("keywords", []) if str(k).strip())

        prompt = (
            "请基于以下非论文资料，提炼最关键的 5 个检索关键词。\\n"
            "要求：\\n"
            "1) 只返回 JSON 数组，例如 [\"关键词1\", \"关键词2\", ...]；\\n"
            "2) 必须返回恰好5个关键词；\\n"
            "3) 关键词尽量专业、可检索；\\n"
            "4) 不要输出已有关键词。\\n\\n"
            f"已有关键词（禁止重复）：{json.dumps(sorted(existing_keywords), ensure_ascii=False)}\\n\\n"
            "资料：\\n"
            + "\\n".join(snippets)
        )

        try:
            response = agent.execute_tool_call("keyword_extraction", {"input": prompt})
            parsed = agent._parse_json(str(response))
            if not isinstance(parsed, list):
                return []

            cleaned: List[str] = []
            seen = set()
            for term in parsed:
                term_text = str(term).strip()
                if not term_text or term_text in existing_keywords or term_text in seen:
                    continue
                seen.add(term_text)
                cleaned.append(term_text)
                if len(cleaned) >= 5:
                    break
            return cleaned
        except Exception as e:
            agent.logger.log(f"⚠️ 非论文关键词总结失败: {e}", level=LogLevel.INFO)
            return []

    @staticmethod
    def extract_initial_concepts(agent, task: str) -> List[Dict[str, Any]]:
        try:
            agent.logger.log("📍 [第1步] 初始概念拆解...", level=LogLevel.INFO)
            skill_input = f"用户研究主题：{task}"
            response = agent.execute_tool_call("initial_concept_decompose", {"input": skill_input})
            concepts = agent._parse_json(str(response))

            if not isinstance(concepts, list) or len(concepts) == 0:
                agent.logger.log("⚠️ 概念拆解结果无效，使用简单分解", level=LogLevel.INFO)
                return [
                    {"concept_name": "核心技术", "keywords": [task]},
                    {"concept_name": "应用场景", "keywords": ["应用", "场景"]},
                ]

            agent.logger.log(f"✅ 成功拆解为 {len(concepts)} 个概念组", level=LogLevel.INFO)
            for i, concept in enumerate(concepts, 1):
                name = concept.get("concept_name", "")
                kws = concept.get("keywords", [])
                agent.logger.log(f"  {i}. {name}: {', '.join(kws[:3])}...", level=LogLevel.INFO)
            return concepts
        except Exception as e:
            agent.logger.log(f"⚠️ 概念拆解失败: {e}，使用简单分解", level=LogLevel.INFO)
            return [
                {"concept_name": "核心技术", "keywords": [task]},
                {"concept_name": "应用场景", "keywords": ["应用", "场景"]},
            ]

    @staticmethod
    def first_shot_retrieval(agent, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            agent.logger.log("🔍 [第2步] First-Shot 初探检索...", level=LogLevel.INFO)

            query_parts = []
            for concept in concepts:
                keywords = concept.get("keywords", [])
                if keywords:
                    group_query = " OR ".join([f'"{kw}"' for kw in keywords])
                    query_parts.append(f"({group_query})")

            search_query = " AND ".join(query_parts)
            agent.logger.log(f"📝 检索式: {search_query}", level=LogLevel.DEBUG)

            if agent.coarse_rag_mode == "web_agent":
                tagged_result = agent._tagged_search_service.run_tagged_web_agent_search(
                    agent,
                    query=search_query,
                    search_topic="First-Shot 概念初探",
                    phase="planning_first_shot",
                    top_k=20,
                )
                payload = tagged_result.get("payload", {})
                items = payload.get("items", []) if isinstance(payload, dict) else []

                papers: List[Dict[str, Any]] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    normalized = KeywordSearchPlanningService._normalize_paper_item(item)
                    if not normalized.get("title") and not normalized.get("abstract"):
                        continue
                    papers.append(normalized)

                plan_citations = tagged_result.get("citations", {})
                if plan_citations:
                    agent._citation_flow_service.add_citations(agent, plan_citations, "规划阶段/First-Shot")

                papers = KeywordSearchPlanningService.enrich_missing_keywords_for_items(agent, papers)
                agent.state[agent.STATE_COARSE_RAG_CONTEXT] = tagged_result.get("context_text", "")
                agent.logger.log(f"✅ tagged web_agent 检索到 {len(papers)} 篇候选资料", level=LogLevel.INFO)
                return papers[:20]

            if agent.coarse_rag_mode == "web_search":
                result = agent.execute_tool_call(agent.search_tool_name, {"query": search_query})
                papers = KeywordSearchPlanningService.parse_search_results_to_papers(str(result))
                papers = KeywordSearchPlanningService.enrich_missing_keywords_for_items(agent, papers)
                agent.logger.log(f"✅ 检索到 {len(papers)} 篇文献", level=LogLevel.INFO)
                return papers[:20]

            agent.logger.log("⏸️ web_search 已禁用，返回空结果", level=LogLevel.INFO)
            return []
        except Exception as e:
            agent.logger.log(f"⚠️ First-Shot 检索失败: {e}", level=LogLevel.INFO)
            return []

    @staticmethod
    def parse_search_results_to_papers(text: str) -> List[Dict[str, Any]]:
        papers = []
        lines = text.split("\n\n")
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.strip().split("\n", 1)
                title = parts[0] if len(parts) > 0 else f"Paper {i+1}"
                abstract = parts[1] if len(parts) > 1 else line.strip()
                papers.append(
                    {
                        "title": title[:200],
                        "abstract": abstract[:1000],
                        "keywords": [],
                        "authors": "Unknown",
                        "url": "N/A",
                        "published_time": "unknown",
                        "year": "",
                        "source_type": "webpage",
                        "apa_citation": "",
                    }
                )
        return papers

    @staticmethod
    def extract_and_filter_keywords(agent, papers: List[Dict[str, Any]], initial_concepts: List[Dict[str, Any]]) -> List[str]:
        try:
            agent.logger.log("📊 [第3步] 术语提取与频次统计...", level=LogLevel.INFO)
            if not papers:
                agent.logger.log("⚠️ 无文献数据，跳过术语提取", level=LogLevel.INFO)
                return []

            paper_lines = []
            for idx, paper in enumerate(papers[:20], 1):
                title = str(paper.get("title", "")).strip()
                abstract = str(paper.get("abstract", "")).strip()
                paper_lines.append(f"[{idx}] Title: {title}")
                paper_lines.append(f"[{idx}] Abstract: {abstract}")
            papers_payload = "\n".join(paper_lines)

            response = agent.execute_tool_call("keyword_extraction", {"input": papers_payload})
            extracted_terms = agent._parse_json(str(response))
            if not isinstance(extracted_terms, list):
                agent.logger.log("⚠️ 术语提取结果无效", level=LogLevel.INFO)
                extracted_terms = []

            from collections import Counter

            term_counter = Counter(extracted_terms)
            existing_keywords = set()
            for concept in initial_concepts:
                existing_keywords.update(concept.get("keywords", []))

            candidate_words = [term for term, count in term_counter.items() if count >= 2 and term not in existing_keywords]

            # 当结果以“非论文来源”为主时，让模型额外总结 5 个关键词补强检索词池
            paper_like_count = sum(1 for paper in papers if KeywordSearchPlanningService._is_paper_like(paper))
            non_paper_majority = paper_like_count < max(1, len(papers) // 2)
            if non_paper_majority:
                fallback_keywords = KeywordSearchPlanningService.summarize_five_keywords_from_non_papers(
                    agent,
                    papers,
                    initial_concepts,
                )
                merged = []
                seen = set()
                for kw in candidate_words + fallback_keywords:
                    kw_text = str(kw).strip()
                    if not kw_text or kw_text in existing_keywords or kw_text in seen:
                        continue
                    seen.add(kw_text)
                    merged.append(kw_text)
                candidate_words = merged
                agent.logger.log(
                    f"ℹ️ 非论文来源占比高，已补充模型总结关键词 {len(fallback_keywords)} 个",
                    level=LogLevel.INFO,
                )

            agent.logger.log(f"✅ 过滤后得到 {len(candidate_words)} 个高频候选词", level=LogLevel.INFO)
            return candidate_words
        except Exception as e:
            agent.logger.log(f"⚠️ 术语提取失败: {e}", level=LogLevel.INFO)
            return []

    @staticmethod
    def upgrade_concepts_with_keywords(
        agent,
        initial_concepts: List[Dict[str, Any]],
        candidate_words: List[str],
    ) -> List[Dict[str, Any]]:
        try:
            agent.logger.log("🔗 [第4步] 语义甄别与概念升级...", level=LogLevel.INFO)
            if not candidate_words:
                agent.logger.log("⚠️ 无候选词，返回原始概念组", level=LogLevel.INFO)
                return initial_concepts

            concepts_lines = []
            for idx, concept in enumerate(initial_concepts, 1):
                concept_name = concept.get("concept_name", f"概念组{idx}")
                keywords = concept.get("keywords", [])
                group_label = chr(ord("A") + idx - 1)
                concepts_lines.append(f"组{group_label}（{concept_name}）：{json.dumps(keywords, ensure_ascii=False)}")

            candidate_hint = json.dumps(candidate_words[:120], ensure_ascii=False)
            skill_input = (
                "目前我们有以下核心概念组：\n"
                + "\n".join(concepts_lines)
                + "\n\n我们在最新文献中发现以下高频候选词：\n"
                + candidate_hint
            )

            response = agent.execute_tool_call("concept_decompose", {"input": skill_input})
            upgraded_concepts = agent._parse_json(str(response))

            if not isinstance(upgraded_concepts, list) or len(upgraded_concepts) == 0:
                agent.logger.log("⚠️ 概念升级失败，返回原始概念组", level=LogLevel.INFO)
                return initial_concepts
            return upgraded_concepts
        except Exception as e:
            agent.logger.log(f"⚠️ 概念升级失败: {e}，返回原始概念组", level=LogLevel.INFO)
            return initial_concepts

    @staticmethod
    def second_shot_retrieval(agent, upgraded_concepts: List[Dict[str, Any]]) -> str:
        try:
            agent.logger.log("🎯 [第5步] Second-Shot 精准检索...", level=LogLevel.INFO)
            query_parts = []
            for concept in upgraded_concepts:
                keywords = concept.get("keywords", [])
                if keywords:
                    group_query = " OR ".join([f'"{kw}"' for kw in keywords])
                    query_parts.append(f"({group_query})")

            search_query = " AND ".join(query_parts)
            if agent.coarse_rag_mode == "web_agent":
                tagged_result = agent._tagged_search_service.run_tagged_web_agent_search(
                    agent,
                    query=search_query,
                    search_topic="Second-Shot 精准收网",
                    phase="planning_second_shot",
                    top_k=12,
                )
                context_text = tagged_result.get("context_text", "")
                plan_citations = tagged_result.get("citations", {})
                if plan_citations:
                    agent._citation_flow_service.add_citations(agent, plan_citations, "规划阶段/Second-Shot")
                agent.state[agent.STATE_COARSE_RAG_CONTEXT] = context_text
                return context_text

            if agent.coarse_rag_mode == "web_search":
                result = agent.execute_tool_call(agent.search_tool_name, {"query": search_query})
                result_text = str(result)
                agent.state[agent.STATE_COARSE_RAG_CONTEXT] = result_text
                return result_text

            return ""
        except Exception as e:
            agent.logger.log(f"⚠️ Second-Shot 检索失败: {e}", level=LogLevel.INFO)
            return ""

    @staticmethod
    def build_outline_generation_input(
        agent,
        task: str,
        upgraded_concepts: List[Dict[str, Any]],
        top_20_papers: List[Dict[str, Any]],
        candidate_keywords: List[str],
        final_retrieval_results: str,
    ) -> str:
        concept_lines = []
        for idx, concept in enumerate(upgraded_concepts, 1):
            name = concept.get("concept_name", "")
            keywords = concept.get("keywords", [])
            concept_lines.append(f"{idx}. {name}: {', '.join(keywords)}")

        candidate_text = ", ".join(candidate_keywords[:30]) if candidate_keywords else "（无）"
        papers_text = KeywordSearchPlanningService.format_paper_materials(agent, top_20_papers, limit=12)
        retrieval_text = agent._trim_text(final_retrieval_results, 5000) or "（无）"

        return (
            f"写作任务：{task}\n\n"
            f"【大纲生成要求】\n"
            f"1. 必须结合下方收集到的资料素材生成详细三级大纲：章、节、子节。\n"
            f"2. 每一级标题都要写成‘核心论点’句式，而不是中性主题词。\n"
            f"3. 每一行都要使用统一的竖线元数据格式：核心论点 | 字数 | 写作目标 | 逻辑关系。\n"
            f"4. 字数要求必须单独写成“| 字数：XXX字”，不要放在标题括号里。\n"
            f"5. 只有叶子节点才真正展开撰写，因此只有叶子节点的字数可以大于0；凡是仍有下级子标题的非叶子节点，字数必须写成“0字”。\n"
            f"6. 如果某章展开到了三级，则该章的一级、二级节点字数都必须为0，只有三级节点字数大于0；如果某章只展开到二级，则一级节点字数为0、二级节点字数大于0。\n"
            f"7. 正文章节重点围绕关键概念展开，避免空泛重复。\n"
            f"8. 需要优先吸收论文标题、摘要与精准检索结果中已经出现的观点、术语、方法和争议。\n\n"
            f"【核心概念组（已升级）】\n"
            f"{'\n'.join(concept_lines) if concept_lines else '（无）'}\n\n"
            f"【候选高频术语】\n{candidate_text}\n\n"
            f"【Top 文献素材】\n{papers_text}\n\n"
            f"【Second-Shot 精准检索结果】\n{retrieval_text}\n"
        )

    @staticmethod
    def format_paper_materials(agent, papers: List[Dict[str, Any]], limit: int = 12) -> str:
        if not papers:
            return "（无）"

        formatted_blocks = []
        for idx, paper in enumerate(papers[:limit], 1):
            title = agent._trim_text(str(paper.get("title", "")).strip(), 220) or f"Paper {idx}"
            abstract = agent._trim_text(str(paper.get("abstract", "")).strip(), 360) or "（无摘要）"
            authors = agent._trim_text(str(paper.get("authors", "")).strip(), 120) or "Unknown"
            url = agent._trim_text(str(paper.get("url", "")).strip(), 220) or "N/A"
            published_time = str(paper.get("published_time", "")).strip() or str(paper.get("year", "")).strip() or "unknown"
            source_type = str(paper.get("source_type", "")).strip() or "webpage"
            formatted_blocks.append(
                f"[{idx}] Title: {title}\n"
                f"Authors: {authors}\n"
                f"Published: {published_time} | SourceType: {source_type}\n"
                f"URL: {url}\n"
                f"Abstract: {abstract}"
            )

        return "\n\n".join(formatted_blocks)


def main() -> int:
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
        print("- build_outline_generation_input(agent, task, upgraded_concepts, top_20_papers, candidate_keywords, final_retrieval_results)")
        print("- format_paper_materials(agent, papers, limit=12)")
        return 0

    print("这是 service 文件，不是可直接执行工作流组件。")
    print("请运行组件文件进行命令行调试，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_keyword_search.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
