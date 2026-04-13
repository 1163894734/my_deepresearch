import argparse
import json
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 引入上下文对象进行类型提示，避免循环导入
    from scripts.multi_agent.agent_context import PipelineContext

class SectionWritingService:

    @staticmethod
    def _json_dumps(data: Any) -> str:
        """Serialize payloads for tool calls while preserving non-ASCII characters."""
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
        """构造章节写作服务使用的 canonical_key。"""
        authors_text = str(authors or "").strip()
        year_text = str(year or "").strip()
        if authors_text and year_text:
            return f"{authors_text} ({year_text})"
        return str(fallback_title or "").strip()

    @staticmethod
    def _unwrap_payload(payload: dict) -> Dict[str, Any]:
        """兼容组件输入的包装结构。"""
        if not isinstance(payload, dict):
            return {}
        inner = payload.get("input")
        if isinstance(inner, dict):
            return inner
        return payload

    @staticmethod
    def _normalize_section(section: Any, default_title: str = "") -> Dict[str, Any]:
        """规范化章节字段并补齐默认值。"""
        section_dict = section if isinstance(section, dict) else {}
        normalized = dict(section_dict)
        if default_title and not str(normalized.get("title", "")).strip():
            normalized["title"] = default_title
        if "word_count_target" in normalized:
            try:
                normalized["word_count_target"] = int(normalized.get("word_count_target") or 0)
            except Exception:
                normalized["word_count_target"] = 0
        else:
            normalized["word_count_target"] = 0
        return normalized

    @staticmethod
    def _normalize_available_citations(raw: Any) -> Dict[str, Dict[str, str]]:
        """规范化组件输入中的引用字典。"""
        if isinstance(raw, dict):
            normalized: Dict[str, Dict[str, str]] = {}
            for k, v in raw.items():
                info = dict(v) if isinstance(v, dict) else {"title": str(v)}
                key = str(k or "").strip()
                title = str(info.get("title") or key).strip()
                if not title:
                    continue
                info["title"] = title
                authors = str(info.get("authors", "")).strip()
                year = str(info.get("year", "")).strip()
                info.setdefault("canonical_key", SectionWritingService._build_canonical_citation_key(authors, year, title))
                normalized[title] = info
            return normalized

        if isinstance(raw, list):
            normalized = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                authors = str(item.get("authors", "")).strip()
                year = str(item.get("year", "")).strip()

                title = str(item.get("title", "")).strip() or "Unknown Citation"
                info = dict(item)
                info["title"] = title
                info.setdefault("canonical_key", SectionWritingService._build_canonical_citation_key(authors, year, title))
                normalized[title] = info
            return normalized

        return {}

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        """独立出来的文本截断纯函数"""
        if not text: 
            return ""
        return text if len(text) <= max_chars else text[:max_chars] + "\n...(已截断)"

    @staticmethod
    def write_intro(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用引言撰写技能并返回文本"""
        return str(context.execute_tool_call(
            "introduction_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def write_conclusion(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用结论撰写技能并返回文本"""
        if "main_text" not in payload and "full_text" in payload:
            payload["main_text"] = payload["full_text"]
            
        return str(context.execute_tool_call(
            "conclusion_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def write_abstract(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用摘要撰写技能并返回文本"""
        if "conclusion_text" not in payload:
            payload["conclusion_text"] = payload.get("conclusion_text") or payload.get("full_text") or payload.get("main_text") or ""
            
        return str(context.execute_tool_call(
            "abstract_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def plan_body_skeleton(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用正文骨架规划技能并返回骨架"""
        return str(context.execute_tool_call(
            "section_skeleton_planning",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def compose_body_content(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责基于骨架调用文本组装技能并返回正文"""
        return str(context.execute_tool_call(
            "section_composition_styling",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def run_fine_rag_web_search(
        context: "PipelineContext",
        search_query: str, 
        search_tool_name: str = "web_search",
        max_chars: int = 1600
    ) -> str:
        """[原子服务] 仅负责执行细粒度网页检索解析并返回摘要"""
        result = context.execute_tool_call(
            search_tool_name,
            {"query": search_query}
        )
        if isinstance(result, str) and result.strip():
            raw_result = result.strip()
            markers = [
                "### 1. Task outcome (short version):",
                '"### 1. Task outcome (short version)":',
                '"final_answer"',
                "tool_code",
            ]
            if any(m in raw_result for m in markers):
                short_match = re.search(
                    r"###\s*1\.\s*Task outcome \(short version\):\s*(.*?)(?:\n###\s*2\.|$)",
                    raw_result,
                    re.IGNORECASE | re.DOTALL,
                )
                if short_match:
                    result = short_match.group(1).strip()
                else:
                    json_match = re.search(
                        r'"###\s*1\.\s*Task outcome \(short version\)"\s*:\s*"(.*?)"\s*,\s*"###\s*2\.',
                        raw_result,
                        re.IGNORECASE | re.DOTALL,
                    )
                    result = json_match.group(1).replace("\\n", "\n").strip() if json_match else raw_result
            else:
                result = raw_result

            return SectionWritingService._trim_text(result, max_chars)
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="SectionWritingService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("SectionWritingService is now a pure atomic tool caller without side-effects.")
        print("\nAvailable public methods:")
        methods = [func for func in dir(SectionWritingService) if callable(getattr(SectionWritingService, func)) and not func.startswith("_")]
        for m in methods:
            print(f" - {m}")
        return 0

    print("这是 service 文件，不直接执行。请在外部主控逻辑（如 Agent 或脚本）中，通过传入 context 对象来调用其静态方法。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())