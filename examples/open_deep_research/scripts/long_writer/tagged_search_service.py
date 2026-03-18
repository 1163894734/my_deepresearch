"""
TaggedSearchService — tagged web-agent search pipeline.

Extracted from LongWriterAgent._parse_tagged_search_json,
_render_tagged_items_as_context, _extract_citations_from_tagged_payload,
_run_tagged_web_agent_search, and _log_tagged_search_record.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from smolagents.monitoring import LogLevel

try:
    from ..citation_validator import (
        build_canonical_citation_key,
        build_preferred_inline_citation,
        extract_author_year_from_key,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from citation_validator import (
        build_canonical_citation_key,
        build_preferred_inline_citation,
        extract_author_year_from_key,
    )

if TYPE_CHECKING:
    pass  # avoid circular imports; agent type is duck-typed


class TaggedSearchService:
    """Stateless service for structured tagged web-agent searches."""

    # ------------------------------------------------------------------ #
    # JSON parsing                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_tagged_search_json(agent, text: str) -> Dict[str, Any]:
        """
        主要作用：从 tagged web agent 输出中稳健抽取 JSON。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """
        raw = str(text or "").strip()
        if not raw:
            return {"query": "", "items": [], "available_citations": {}}

        # provide_run_summary=True 会在正文后追加 <summary_of_work>…</summary_of_work>，
        # 其中包含大量浏览步骤的 JSON 片段，必须在解析前截断。
        summary_cut = re.search(r"<summary_of_work>", raw, re.IGNORECASE)
        if summary_cut:
            raw = raw[: summary_cut.start()].strip()

        # 同理截掉 managed_agent report 包装的前缀（"Here is the final answer from..."）
        report_match = re.search(
            r"Here is the final answer from your managed agent ['\"]?[\w_-]+['\"]?:\s*",
            raw, re.IGNORECASE
        )
        if report_match:
            raw = raw[report_match.end():].strip()

        fenced = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1).strip()

        try:
            data = json.loads(raw)
        except Exception:
            obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if obj_match:
                try:
                    data = json.loads(obj_match.group(0))
                except Exception:
                    data = None
            else:
                data = None

        if isinstance(data, list):
            data = {"query": "", "items": data}

        if not isinstance(data, dict):
            return {"query": "", "items": [], "available_citations": {}}

        items = data.get("items", [])
        if not isinstance(items, list):
            items = []

        normalized_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_items.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "source_type": str(item.get("source_type", "webpage")).strip() or "webpage",
                    "time": str(item.get("time", "unknown")).strip() or "unknown",
                    "direction": str(item.get("direction", "unknown")).strip() or "unknown",
                    "content_summary": str(item.get("content_summary", "")).strip(),
                    "existing_problems": item.get("existing_problems", []) if isinstance(item.get("existing_problems", []), list) else [],
                    "foundation": str(item.get("foundation", "")).strip(),
                    "outlook": str(item.get("outlook", "")).strip(),
                    "future_directions": item.get("future_directions", []) if isinstance(item.get("future_directions", []), list) else [],
                    "relevance_score": item.get("relevance_score", 0),
                    "evidence_quote": str(item.get("evidence_quote", "")).strip(),
                    "authors": str(item.get("authors", "")).strip(),
                    "year": str(item.get("year", "")).strip(),
                    "apa_citation": str(item.get("apa_citation", "")).strip(),
                }
            )

        # 统一格式：available_citations 必须是 Dict["title", citation_info]
        normalized_citations: Dict[str, Dict[str, Any]] = {}
        raw_citations = data.get("available_citations", {})
        if isinstance(raw_citations, dict):
            for k, v in raw_citations.items():
                info = dict(v) if isinstance(v, dict) else {"title": str(v)}
                title = str(info.get("title") or k).strip()
                if not title:
                    continue
                info["title"] = title
                authors = str(info.get("authors", "")).strip()
                year = str(info.get("year", "")).strip()
                canonical_key = build_canonical_citation_key(authors, year, title)
                if canonical_key:
                    info["canonical_key"] = canonical_key
                normalized_citations[title] = info

        # 若模型未显式输出 available_citations，则从 items 回填（兼容旧输出）
        if not normalized_citations:
            for item in normalized_items:
                authors = str(item.get("authors", "")).strip()
                year = str(item.get("year", "")).strip()
                apa = str(item.get("apa_citation", "")).strip()
                key = ""
                if authors and year:
                    key = str(item.get("title", "")).strip()
                elif apa:
                    apa_match = re.match(r"^\(([^()]{1,120}?),\s*((?:19|20)\d{2}|n\.d\.)\)$", apa)
                    if apa_match:
                        key = str(item.get("title", "")).strip()
                if not key:
                    continue
                canonical_key = build_canonical_citation_key(authors, year, key)
                normalized_citations[key] = {
                    "authors": authors,
                    "year": year,
                    "title": str(item.get("title", "")).strip() or key,
                    "url": str(item.get("url", "")).strip(),
                    "source_type": str(item.get("source_type", "webpage")).strip() or "webpage",
                    "apa_citation": apa,
                    "canonical_key": canonical_key,
                }

        return {
            "query": str(data.get("query", "")).strip(),
            "items": normalized_items,
            "available_citations": normalized_citations,
        }

    # ------------------------------------------------------------------ #
    # Context rendering                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def render_tagged_items_as_context(agent, payload: Dict[str, Any], limit: int = 1800) -> str:
        """
        主要作用：把结构化检索结果渲染成写作可读上下文。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (Dict[str, Any]): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。
        - limit (int): 文本截断长度或输出上限。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            return ""

        blocks = []
        for idx, item in enumerate(items, 1):
            title = str(item.get("title", "")).strip() or f"Untitled Source {idx}"
            url = str(item.get("url", "")).strip() or "N/A"
            direction = str(item.get("direction", "unknown")).strip()
            time_tag = str(item.get("time", "unknown")).strip()
            summary = str(item.get("content_summary", "")).strip() or "（无摘要）"
            problems = item.get("existing_problems", []) if isinstance(item.get("existing_problems", []), list) else []
            foundation = str(item.get("foundation", "")).strip() or "（无）"
            outlook = str(item.get("outlook", "")).strip() or "（无）"
            future = item.get("future_directions", []) if isinstance(item.get("future_directions", []), list) else []
            quote = str(item.get("evidence_quote", "")).strip()
            apa = str(item.get("apa_citation", "")).strip()

            blocks.append(
                f"[{idx}] {title}\n"
                f"URL: {url}\n"
                f"方向: {direction} | 时间: {time_tag}\n"
                f"简短总结: {summary}\n"
                f"现有问题: {'; '.join(problems) if problems else '（无）'}\n"
                f"基础: {foundation}\n"
                f"展望: {outlook}\n"
                f"未来方向: {'; '.join(future) if future else '（无）'}\n"
                f"证据摘录: {quote or '（无）'}\n"
                f"APA文内引用: {apa or '（无）'}"
            )

        return agent._trim_text("\n\n".join(blocks), limit)

    # ------------------------------------------------------------------ #
    # Citation extraction from tagged payload                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_citations_from_tagged_payload(agent, payload: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        主要作用：从 tagged 搜索结果中提取标题索引引用字典。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (Dict[str, Any]): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - Dict[str, Dict[str, str]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """
        citations: Dict[str, Dict[str, str]] = {}

        # 新标准：优先读取 available_citations（Dict）
        if isinstance(payload, dict):
            raw_citations = payload.get("available_citations", {})
            if isinstance(raw_citations, dict) and raw_citations:
                for key, info in raw_citations.items():
                    if not isinstance(info, dict):
                        continue
                    authors = str(info.get("authors", "")).strip()
                    year = str(info.get("year", "")).strip()
                    title = str(info.get("title") or key).strip()
                    url = str(info.get("url", "")).strip()
                    inline_citation = str(info.get("apa_citation", "")).strip() or str(info.get("inline_citation", "")).strip()
                    if not inline_citation and authors and year:
                        inline_citation = build_preferred_inline_citation(authors, year)

                    norm_key = str(info.get("canonical_key") or "").strip()
                    if (not authors or not year) and norm_key:
                        k_author, k_year = extract_author_year_from_key(norm_key)
                        authors = authors or k_author
                        year = year or k_year

                    if not title or not authors or not year:
                        continue

                    canonical_key = build_canonical_citation_key(authors, year, title)
                    citations[title] = {
                        "authors": authors,
                        "year": year,
                        "title": title or "Retrieved from tagged web agent",
                        "url": url,
                        "title_source": "tagged_web_agent",
                        "inline_citation": inline_citation,
                        "canonical_key": canonical_key,
                    }
                return citations

        # 兼容旧格式：从 items（List）提取
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            return citations

        placeholder_authors = {
            "anonymous", "anonymous authors", "author", "authors", "unknown", "n/a", "佚名", "作者", "作者等"
        }

        for item in items:
            if not isinstance(item, dict):
                continue
            authors = str(item.get("authors", "")).strip()
            year = str(item.get("year", "")).strip()
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            apa_citation = str(item.get("apa_citation", "")).strip()

            # 将占位作者视为缺失，避免把 Anonymous 写入参考文献
            if authors.lower() in placeholder_authors:
                authors = ""

            # 从 APA 文内引用中回填作者与年份，如：(van Baalen et al., 2023)
            if (not authors or not year) and apa_citation:
                apa_match = re.match(r"^\(([^()]{1,120}?),\s*((?:19|20)\d{2}|n\.d\.)\)$", apa_citation)
                if apa_match:
                    apa_author = apa_match.group(1).strip()
                    apa_year = apa_match.group(2).strip()
                    if not authors and apa_author.lower() not in placeholder_authors:
                        authors = apa_author
                    if not year:
                        year = apa_year

                # 兜底：只回填年份
                if not year:
                    year_match = re.search(r"\((?:[^()]*?,\s*)?((?:19|20)\d{2}|n\.d\.)\)", apa_citation)
                    if year_match:
                        year = year_match.group(1)

            if not authors or not year:
                continue

            inline_citation = (
                apa_citation
                if re.match(r"^\(.+?,\s*((?:19|20)\d{2}|n\.d\.)\)$", apa_citation)
                else build_preferred_inline_citation(authors, year)
            )

            key = title or build_canonical_citation_key(authors, year, title)
            citations[key] = {
                "authors": authors,
                "year": year,
                "title": title or "Retrieved from tagged web agent",
                "url": url,
                "title_source": "tagged_web_agent",
                "inline_citation": inline_citation,
                "canonical_key": build_canonical_citation_key(authors, year, title),
            }

        return citations

    # ------------------------------------------------------------------ #
    # Orchestration                                                        #
    # ------------------------------------------------------------------ #

    # JSON schema 片段，用于 LLM 提取时的提示
    _TAGGED_ITEM_SCHEMA = """
{
  "query": "...",
    "available_citations": {
        "Paper Title": {
            "authors": "",
            "year": "",
            "title": "",
            "url": "",
            "source_type": "webpage|paper|report|news|other",
            "apa_citation": "(Author, Year)",
            "canonical_key": "Author (Year)"
        }
    },
  "items": [
    {
      "title": "",
      "url": "",
      "source_type": "webpage|paper|report|news|other",
      "time": "YYYY 或 YYYY-MM 或 unknown",
      "direction": "",
      "content_summary": "",
      "existing_problems": [""],
      "foundation": "",
      "outlook": "",
      "key_words": [""],
      "future_directions": [""],
      "relevance_score": 0,
      "evidence_quote": "",
      "authors": "",
      "year": "",
      "apa_citation": "(Author, Year)"
    }
  ]
}
"""

    @staticmethod
    def _extract_json_via_llm(agent, query: str, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        主要作用：在原始输出不规范时借助模型二次抽取 JSON。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - query (str): 检索查询文本。
        - raw_text (str): 该参数用于承载 `raw_text` 相关的业务上下文或控制信息。

        返回值：
        - Optional[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """
        try:
            from smolagents.models import ChatMessage, MessageRole  # 延迟导入避免循环
        except ImportError:
            return None

        snippet = raw_text[:6000]
        prompt = (
            f"以下是一个网页搜索 agent 的输出文本，请将其中的搜索结果提取整理为如下 JSON schema，"
            f"只输出合法 JSON，不要任何解释文字。\n\n"
            f"查询: {query}\n\n"
            f"目标 schema:\n{TaggedSearchService._TAGGED_ITEM_SCHEMA}\n\n"
            f"原始文本（截取前6000字符）:\n{snippet}"
        )
        messages = [
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        try:
            response = agent.model(messages)
            resp_text = response.content if hasattr(response, "content") else str(response)
            payload = TaggedSearchService.parse_tagged_search_json(agent, resp_text)
            if payload.get("items"):
                return payload
        except Exception as exc:
            agent.logger.log(f"⚠️ LLM 结构化提取异常: {exc}", level=LogLevel.INFO)
        return None

    @staticmethod
    def run_tagged_web_agent_search(
        agent,
        query: str,
        top_k: int = 8,
        search_time: Optional[str] = "2024-2026年",
    ) -> Dict[str, Any]:
        """
        主要作用：调用 tagged web agent 执行结构化网页检索。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - query (str): 检索查询文本。
        - top_k (int): 希望保留的搜索结果上限。
        - search_time (Optional[str]): 检索时使用的时间范围提示。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """
        task_prompt = f"""请围绕如下查询进行检索，并仅输出合法 JSON：

查询: {query}
时间: {search_time}
要求:
1. 搜索并访问最相关网页/论文，筛选前 {max(1, top_k)} 条最有价值的信息源；
2. 对每条信息源进行简短总结，并打标签（time, direction, existing_problems, foundation, outlook, future_directions）；
3. 输出中必须包含 available_citations（Dict），键必须是文献标题 title；
4. 每条都尽量提供可追溯引用信息（authors, year, title, url, apa_citation）；
5. items 与 available_citations 要一致（同一来源应在两处都体现）；
6. authors 必须是真实作者姓名；严禁填写 Anonymous / Anonymous Authors / unknown / 佚名 等占位词；
7. 若来源为 arXiv，请优先从论文页面提取作者并写入 authors 字段；
8. 只输出 JSON，不要解释文字。

JSON schema:
{{
  "query": "...",
    "available_citations": {{
        "Paper Title1": {{
            "authors": "",
            "year": "",
            "title": "",
            "url": "",
            "source_type": "webpage|paper|report|news|other",
            "apa_citation": "(Author, Year)",
            "canonical_key": "Author (Year)"
        }},
        "Paper Title2": {{
            "authors": "",
            "year": "",
            "title": "",
            "url": "",
            "source_type": "webpage|paper|report|news|other",
            "apa_citation": "(Author, Year)",
            "canonical_key": "Author (Year)"
        }}
    }}
}}
"""

        raw_result = agent.execute_tool_call(agent.text_webbrowser_agent_name, {"task": task_prompt})
        cleaned = agent._clean_agent_output(str(raw_result))
        payload = TaggedSearchService.parse_tagged_search_json(agent, cleaned)
        payload_items = payload.get("items", []) if isinstance(payload, dict) else []
        if not payload_items and cleaned:
            agent.logger.log(
                "⚠️ tagged 检索输出非结构化 JSON，启用 LLM 结构化提取...",
                level=LogLevel.INFO,
            )
            llm_payload = TaggedSearchService._extract_json_via_llm(agent, query, cleaned)
            if llm_payload and llm_payload.get("items"):
                agent.logger.log(
                    f"✅ LLM 结构化提取成功，共 {len(llm_payload['items'])} 条",
                    level=LogLevel.INFO,
                )
                payload = llm_payload
            else:
                agent.logger.log(
                    "⚠️ LLM 提取仍无有效 items，保持空结果",
                    level=LogLevel.INFO,
                )
                payload = {"query": str(query or ""), "items": []}

        context_text = TaggedSearchService.render_tagged_items_as_context(agent, payload, limit=agent.max_fine_rag_chars)
        citations = TaggedSearchService.extract_citations_from_tagged_payload(agent, payload)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "search_time": search_time,
            "payload": payload,
            "context_preview": agent._trim_text(context_text, 800),
            "citations_count": len(citations),
        }
        agent._tagged_retrieval_records.append(record)
        agent.state[agent.STATE_TAGGED_RETRIEVAL_RECORDS] = agent._tagged_retrieval_records
        agent.state[agent.STATE_LATEST_TAGGED_SEARCH_PAYLOAD] = payload
        TaggedSearchService.log_tagged_search_record(agent, record)

        return {
            "raw": str(raw_result),
            "cleaned": cleaned,
            "payload": payload,
            "context_text": context_text,
            "citations": citations,
        }

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def log_tagged_search_record(agent, record: Dict[str, Any]) -> None:
        """
        主要作用：记录 tagged 检索的输入、输出与摘要。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - record (Dict[str, Any]): 已验证的引用记录对象，包含标题、作者、摘要和验证状态等信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """
        try:
            # 1) JSONL 结构化日志（完整）
            with open(agent._tagged_search_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 2) generation_log 摘要（便于人眼快速查看）
            with open(agent._log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"【Tagged检索】{record.get('timestamp', '')}\n")
                f.write(f"phase: {record.get('phase', '')}\n")
                f.write(f"topic: {record.get('topic', '')}\n")
                f.write(f"query: {record.get('query', '')}\n")
                payload_items = (record.get('payload') or {}).get('items', []) if isinstance(record.get('payload'), dict) else []
                f.write(f"items_count: {len(payload_items)}\n")
                f.write(f"citations_count: {record.get('citations_count', 0)}\n")
                f.write("context_preview:\n")
                f.write(str(record.get('context_preview', '')) + "\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            agent.logger.log(f"⚠️ 写入 tagged 检索日志失败: {e}")


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

    parser = argparse.ArgumentParser(description="TaggedSearchService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("TaggedSearchService static methods:")
        print("- parse_tagged_search_json(agent, text)")
        print("- render_tagged_items_as_context(agent, payload, limit=1800)")
        print("- extract_citations_from_tagged_payload(agent, payload)")
        print("- run_tagged_web_agent_search(agent, query, top_k=8)")
        print("- log_tagged_search_record(agent, record)")
        return 0

    print("这是 service 文件，不直接执行。")
    print("请调试组件文件，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_keyword_search.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
