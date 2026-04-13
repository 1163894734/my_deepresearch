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
from utils.common_utils import execute_tool_call

if TYPE_CHECKING:
    pass  # avoid circular imports; agent type is duck-typed


class TaggedSearchService:
    """Stateless service for structured tagged web-agent searches."""

    @staticmethod
    def _extract_json_candidate(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if text.startswith("```"):
            fence_match = re.search(r"```(?:json)?\\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
            if fence_match:
                text = fence_match.group(1).strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0).strip() if match else text

    @staticmethod
    def _escape_invalid_backslashes(text: str) -> str:
        # Keep legal JSON escapes intact; only escape stray backslashes such as "\Delta".
        return re.sub(r"(?<!\\\\)\\\\(?![\\\\\"/bfnrtu])", r"\\\\\\\\", text)

    @staticmethod
    def _parse_tagged_result_json(raw_result: Any) -> Dict[str, Any]:
        if isinstance(raw_result, dict):
            return raw_result

        candidate = TaggedSearchService._extract_json_candidate(str(raw_result))
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            repaired = TaggedSearchService._escape_invalid_backslashes(candidate)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                raise ValueError(f"tagged 检索结果不是合法 JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError("tagged 检索结果 JSON 顶层必须是对象")
        return parsed

    # ------------------------------------------------------------------ #
    # JSON parsing                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def run_tagged_web_agent_search(
        agent,
        query: str,
        top_k: int = 20,
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
5. authors 必须是真实作者姓名；严禁填写 Anonymous / Anonymous Authors / unknown / 佚名 等占位词；
6. 若来源为 arXiv，请优先从论文页面提取作者并写入 authors 字段；
7. 只输出 JSON，不要解释文字,严格遵循以下的json格式输出。
8. 输出的 available_citations 中的文献必须与查询高度相关，且优先包含2024-2026年发表的文献。
9. key_words优先从论文的关键词中提取，如果没有，可以从标题和摘要中提取，确保关键词具有代表性和区分度。

JSON schema:
{{
  "query": "大模型领域前沿进展",
    "available_citations": {{
        "Paper Title1": {{
            "authors": "",
            "year": "",
            "title": "",
            "url": "",
            "source_type": "webpage|paper|report|news|other",
            "apa_citation": "(Author, Year)",
            "key_words": [""],
            "abstract": ""
        }},
        "Paper Title2": {{
            "authors": "",
            "year": "",
            "title": "",
            "url": "",
            "source_type": "webpage|paper|report|news|other",
            "apa_citation": "(Author, Year)",
            "key_words": [""],
            "abstract": ""
        }}
    }}
}}
"""

        available_tools = {**getattr(agent, "tools", {}), **getattr(agent, "managed_agents", {})}
        raw_result = execute_tool_call(
            agent.text_webbrowser_agent_name,
            {"task": task_prompt},
            available_tools=available_tools,
            logger=getattr(agent, "logger", None),
        )
        agent.logger.log(
            "✅raw_result:\n"+raw_result,
            level=LogLevel.INFO,
        )
        try:
            return TaggedSearchService._parse_tagged_result_json(raw_result)
        except Exception as e:
            agent.logger.log(f"⚠️ tagged 结果 JSON 解析失败: {e}", level=LogLevel.ERROR)
            raise

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

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
