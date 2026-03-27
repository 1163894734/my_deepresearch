from __future__ import annotations

import argparse
import json
import re
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    try:
        from ..long_writer_agent_v3 import LongWriterAgent
    except Exception:
        from long_writer_agent_v3 import LongWriterAgent

from smolagents.monitoring import LogLevel

try:
    from .outline_parsing_service import OutlineParsingService
except ImportError:
    from outline_parsing_service import OutlineParsingService

class SectionWritingService:

    @staticmethod
    def _json_dumps(data: Any) -> str:
        """Serialize payloads for tool calls while preserving non-ASCII characters."""
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _try_parse_json(text: Any) -> Any:
        """Parse JSON text when possible; otherwise return the original value."""
        if not isinstance(text, str):
            return text
        raw = text.strip()
        if not raw:
            return text
        try:
            return json.loads(raw)
        except Exception:
            return text

    @staticmethod
    def _build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
        """
        主要作用：构造章节写作服务使用的 canonical_key。
        """
        authors_text = str(authors or "").strip()
        year_text = str(year or "").strip()
        if authors_text and year_text:
            return f"{authors_text} ({year_text})"
        return str(fallback_title or "").strip()

    @staticmethod
    def _unwrap_payload(payload: dict) -> Dict[str, Any]:
        """
        主要作用：兼容组件输入的包装结构。
        """
        if not isinstance(payload, dict):
            return {}
        inner = payload.get("input")
        if isinstance(inner, dict):
            return inner
        return payload

    @staticmethod
    def _normalize_section(section: Any, default_title: str = "") -> Dict[str, Any]:
        """
        主要作用：规范化章节字段并补齐默认值。
        """
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
        """
        主要作用：规范化组件输入中的引用字典。
        """
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
                apa = str(item.get("apa_citation", "")).strip()
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
    def write_intro_content_with_json(agent: "LongWriterAgent", payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成引言章节内容。（核心逻辑）
        """
        try:
            # 直接把 payload 丢给工具（里面已经有 section, available_citations, outline 等了）
            content = agent.execute_tool_call("introduction_write", {"input": SectionWritingService._json_dumps(payload)})
            agent.logger.log("✅ 引言撰写完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 引言撰写失败: {e}", level=LogLevel.ERROR)
            raise
            
        section = payload.get("section", {})
        available_citations = payload.get("available_citations", {})
        
        available_citations, content = agent._citation_validator.run_five_step_validation(
            available_citations,
            content,
            log_file_path=getattr(agent, "_citations_validation_log", None),
            section_ref=str(section.get("title", "引言")),
        )
        return content

    @staticmethod
    def write_intro_content(
        agent: "LongWriterAgent",
        section: Dict[str, Any],
        available_citations: Dict[str, Dict[str, str]],
        outline: str = "",
    ) -> str:
        """
        主要作用：撰写引言。（兼容保留层）
        """
        payload = {
            "section": section,
            "available_citations": available_citations,
            "outline": outline
        }
        return SectionWritingService.write_intro_content_with_json(agent, payload)

    @staticmethod
    def write_conclusion_content_with_json(agent: "LongWriterAgent", payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成结论章节内容。（核心逻辑）
        """
        # 兼容一下键名：Agent传的是 full_text，工具可能期望 main_text
        if "main_text" not in payload and "full_text" in payload:
            payload["main_text"] = payload["full_text"]
            
        try:
            content = agent.execute_tool_call("conclusion_write", {"input": SectionWritingService._json_dumps(payload)})
            agent.logger.log("✅ 结论撰写完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 结论撰写失败: {e}", level=LogLevel.ERROR)
            raise
            
        section = payload.get("section", {})
        available_citations = payload.get("available_citations", {})
        
        available_citations, content = agent._citation_validator.run_five_step_validation(
            available_citations,
            content,
            log_file_path=getattr(agent, "_citations_validation_log", None),
            section_ref=str(section.get("title", "结论")),
        )
        return content

    @staticmethod
    def write_conclusion_content(
        agent: "LongWriterAgent",
        section: Dict[str, Any],
        main_text: str,
        available_citations: Dict[str, Dict[str, str]],
        outline: str = "",
    ) -> str:
        """
        主要作用：撰写结论。（兼容保留层）
        """
        payload = {
            "section": section,
            "main_text": main_text,
            "available_citations": available_citations,
            "outline": outline
        }
        return SectionWritingService.write_conclusion_content_with_json(agent, payload)
    @staticmethod
    def write_abstract_content_with_json(agent: "LongWriterAgent", payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成摘要章节内容。（核心逻辑）
        """
        # 补齐工具需要的 conclusion_text
        if "conclusion_text" not in payload:
            payload["conclusion_text"] = payload.get("conclusion_text") or payload.get("full_text") or payload.get("main_text") or ""
            
        try:
            content = agent.execute_tool_call("abstract_write", {"input": SectionWritingService._json_dumps(payload)})
            return str(content)
        except Exception as e:
            agent.logger.log(f"⚠️ 摘要生成失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def write_abstract_section(
        agent: "LongWriterAgent", 
        section: Dict[str, str], 
        conclusion_text: str, 
        outline: str = ""
    ) -> str:
        """
        主要作用：直接调用摘要技能生成摘要文本。（兼容保留层）
        """
        payload = {
            "section": section,
            "conclusion_text": conclusion_text,
            "outline": outline
        }
        return SectionWritingService.write_abstract_content_with_json(agent, payload)

    @staticmethod
    def write_body_content_with_json(agent: "LongWriterAgent", payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成正文章节内容。
        """
        # 直接从 payload 中提取需要的参数
        section = payload.get("section", {})
        fine_rag_context = payload.get("fine_rag_context", "")
        available_citations = payload.get("available_citations", {})
        
        # 直接调用 write_body_content
        return SectionWritingService.write_body_content(
            agent=agent,
            section=section,
            fine_rag_context=fine_rag_context,
            available_citations=available_citations
        )

    @staticmethod
    def run_fine_rag_web_search(agent: "LongWriterAgent", section_title: str, search_query: str) -> str:
        """
        主要作用：执行细粒度网页检索。
        """
        try:
            agent.logger.log("🔎 调用DuckDuckGo搜索", level=LogLevel.DEBUG)
            result = agent.execute_tool_call(agent.search_tool_name, {"query": search_query})
            if isinstance(result, str) and result.strip():
                raw_result = result.strip()
                markers = [
                    "### 1. Task outcome (short version):",
                    '"### 1. Task outcome (short version)":',
                    '"final_answer"',
                    "```tool_code",
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

                result_len = len(result)
                agent.logger.log(f"✅ web_search成功 ({result_len} 字)", level=LogLevel.DEBUG)
                return agent._trim_text(result, agent.max_fine_rag_chars)
        except Exception as e:
            agent.logger.log(f"  ⚠️ web_search调用失败: {e}", level=LogLevel.DEBUG)
            raise


    @staticmethod
    def write_body_content_with_json(agent: "LongWriterAgent", payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成正文章节内容。（核心逻辑）
        """
        section = payload.get("section", {})
        available_citations = payload.get("available_citations", {})
        
        try:
            # 骨架规划
            skeleton = agent.execute_tool_call("section_skeleton_planning", {"input": SectionWritingService._json_dumps(payload)})
            agent.logger.log("✅ 骨架规划完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 骨架规划失败: {e}", level=LogLevel.ERROR)
            raise

        try:
            # 文本组装：直接在 payload 基础上加入 skeleton 传给下一个工具
            composition_payload = dict(payload)
            composition_payload["skeleton"] = skeleton
            content = agent.execute_tool_call("section_composition_styling", {"input": SectionWritingService._json_dumps(composition_payload)})
            agent.logger.log("✅ 文本组装完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 文本组装失败: {e}", level=LogLevel.ERROR)
            raise
            
        available_citations, content = agent._citation_validator.run_five_step_validation(
            available_citations,
            content,
            log_file_path=getattr(agent, "_citations_validation_log", None),
            section_ref=str(section.get("title", "正文")),
        )
        return str(SectionWritingService.section_reflection_loop(agent, content, section, available_citations))

    @staticmethod
    def write_body_content(
        agent: "LongWriterAgent",
        section: Dict[str, Any],
        fine_rag_context: str,
        available_citations: Dict[str, Dict[str, str]],
        outline: str = "",
    ) -> str:
        """
        主要作用：执行正文章节的骨架规划与成稿。（兼容保留层）
        """
        payload = {
            "section": section,
            "fine_rag_context": fine_rag_context,
            "available_citations": available_citations,
            "outline": outline
        }
        return SectionWritingService.write_body_content_with_json(agent, payload)

    @staticmethod
    def section_reflection_loop(
        agent: "LongWriterAgent",
        text: str,
        section: Dict[str, str],
        available_citations: Dict = None,
    ) -> str:
        """
        主要作用：执行正文章节的反思—修订循环。
        """
        current = text

        try:
            # 1. 反思：调用 evidence_reflection skill，由大模型判断质量
            agent.logger.log("  🔍 证据反思...", level=LogLevel.DEBUG)
            evidence_reflection_input = {"section": section, "content": current, "available_citations": available_citations}
            report_raw = agent.execute_tool_call("evidence_reflection", {"input": SectionWritingService._json_dumps(evidence_reflection_input)})
            report = str(report_raw)
            agent._log_reflection_to_file(f"证据反思 - {section.get('goal', '')[:30]}", 0, report)

            # 2. 修订：将原文 + 反思报告交给 section_revision skill，由大模型直接输出修订稿
            section_revision_input = {
                "section": section,
                "original_content": current,
                "reflection_report": SectionWritingService._try_parse_json(report),
            }
            current = str(agent.execute_tool_call("section_revision", {"input": SectionWritingService._json_dumps(section_revision_input)}))
            agent._log_revision_to_file(f"段落修订 - {section.get('goal', '')[:30]}", current)
            agent.logger.log("  ✅ 段落修订完成", level=LogLevel.DEBUG)

        except Exception as e:
            agent.logger.log(f"  ⚠️ 反思出错: {e}", level=LogLevel.ERROR)
            raise

        return current

def main() -> int:
    """
    主要作用：执行 main 相关逻辑。
    """
    parser = argparse.ArgumentParser(description="SectionWritingService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("SectionWritingService: use component files for end-to-end CLI debugging.")
        return 0

    print("这是 service 文件，不直接执行。")
    print("请调试组件文件，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_body.py --help")
    print("python examples/open_deep_research/scripts/long_writer/component_introduction.py --help")
    print("python examples/open_deep_research/scripts/long_writer/component_conclusion.py --help")
    print("python examples/open_deep_research/scripts/long_writer/component_abstract.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())