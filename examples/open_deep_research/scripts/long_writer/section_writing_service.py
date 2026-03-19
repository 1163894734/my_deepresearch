from __future__ import annotations

import argparse
import re
from typing import Any, Dict

from smolagents.monitoring import LogLevel

try:
    from .outline_parsing_service import OutlineParsingService
except ImportError:
    from outline_parsing_service import OutlineParsingService

try:
    from ..citation_validator import format_allowed_citation_whitelist
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from citation_validator import format_allowed_citation_whitelist


class SectionWritingService:

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
    def write_intro_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成引言章节内容。
        """
        # 直接从 payload 中提取需要的参数
        section = payload.get("section", {})
        word_count_target = int(section.get("word_count_target", 0) or 0)
        available_citations = payload.get("available_citations", {})
        
        # 构建 fine_rag_context（如果需要）
        fine_rag_context = payload.get("fine_rag_context", "")
        
        # 直接调用对应的写作方法（假设存在 write_intro_content 方法）
        # 如果不存在，需要先创建对应的方法
        return SectionWritingService.write_intro_content(
            agent=agent,
            section=section,
            fine_rag_context=fine_rag_context,
            available_citations=available_citations
        )

    @staticmethod
    def write_conclusion_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成结论章节内容。
        """
        # 直接从 payload 中提取需要的参数
        section = payload.get("section", {})
        word_count_target = int(section.get("word_count_target", 0) or 0)
        available_citations = payload.get("available_citations", {})
        
        # 构建 main_text（如果需要）
        main_text = payload.get("main_text", "")
        
        # 直接调用对应的写作方法（假设存在 write_conclusion_content 方法）
        return SectionWritingService.write_conclusion_content(
            agent=agent,
            section=section,
            main_text=main_text,
            available_citations=available_citations
        )

    @staticmethod
    def write_abstract_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成摘要章节内容。
        """
        # 直接从 payload 中提取需要的参数
        section = payload.get("section", {})
        
        # 获取结论文本，支持多种字段名
        conclusion_text = (
            payload.get("conclusion_text") or 
            payload.get("full_text") or 
            payload.get("main_text") or 
            ""
        )
        
        # 直接调用 write_abstract_section
        return SectionWritingService.write_abstract_section(
            agent=agent,
            section=section,
            conclusion_text=conclusion_text
        )

    @staticmethod
    def write_body_content_with_json(agent, payload: dict) -> str:
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
    def run_fine_rag_web_search(agent, section_title: str, search_query: str) -> str:
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
    def write_abstract_section(agent, section: Dict[str, str], conclusion_text: str) -> str:
        """
        主要作用：直接调用摘要技能生成摘要文本。
        """
        try:
            abstract_input = SectionWritingService.build_abstract_input(agent, section, conclusion_text)
            content = agent.execute_tool_call("abstract_write", {"input": abstract_input})
            return str(content)
        except Exception as e:
            agent.logger.log(f"⚠️ 摘要生成失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def write_body_content(
        agent,
        section: Dict[str, Any],
        fine_rag_context: str,
        available_citations: Dict[str, Dict[str, str]],
    ) -> str:
        """
        主要作用：执行正文章节的骨架规划与成稿。
        """
        word_count_target = int(section.get("word_count_target", 0) or 0)

        skeleton_input = SectionWritingService.build_skeleton_planning_input(agent, section, fine_rag_context)
        try:
            skeleton = agent.execute_tool_call("section_skeleton_planning", {"input": skeleton_input})
            agent.logger.log("✅ 骨架规划完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 骨架规划失败: {e}", level=LogLevel.ERROR)
            raise

        section_with_citations = dict(section)
        section_with_citations["_available_citations"] = available_citations
        composition_input = SectionWritingService.build_composition_input(
            agent,
            section_with_citations,
            fine_rag_context,
            skeleton,
            word_count_target,
        )
        try:
            content = agent.execute_tool_call("section_composition_styling", {"input": composition_input})
            agent.logger.log("✅ 文本组装完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 文本组装失败: {e}", level=LogLevel.ERROR)
            raise

        return str(SectionWritingService.section_reflection_loop(agent, content, section, word_count_target, available_citations))

    @staticmethod
    def section_reflection_loop(
        agent,
        text: str,
        section: Dict[str, str],
        word_count_target: int = 0,
        available_citations: Dict = None,
    ) -> str:
        """
        主要作用：执行正文章节的反思—修订循环。
        """
        current = text
        section_title = section.get("title", "未命名章节")
        goal = section.get("goal", "")

        try:
            # 1. 反思：调用 evidence_reflection skill，由大模型判断质量
            agent.logger.log("  🔍 证据反思...", level=LogLevel.DEBUG)
            reflection_input = (
                f"章节标题：{section_title}\n"
                f"章节目标：{goal}\n\n"
                f"内容：\n{current}"
            )
            report = str(agent.execute_tool_call("evidence_reflection", {"input": reflection_input}))
            agent._log_reflection_to_file(f"证据反思 - {goal[:30]}", 0, report)

            # 2. 修订：将原文 + 反思报告交给 section_revision skill，由大模型直接输出修订稿
            revision_prompt = (
                f"章节标题：{section_title}\n\n"
                f"原始内容：\n{current}\n\n"
                f"反思报告：\n{report}\n\n"
                "请根据反思报告修订内容，直接输出修订后的完整段落，不要输出任何解释。"
            )
            current = str(agent.execute_tool_call("section_revision", {"input": revision_prompt}))
            agent._log_revision_to_file(f"段落修订 - {goal[:30]}", current)
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