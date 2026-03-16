from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Tuple

from smolagents.monitoring import LogLevel


class OutlineParsingService:
    """大纲解析服务：章节抽取、叶子节点标注与写作前清洗。"""

    @staticmethod
    def parse_sections(agent, outline: str) -> List[Dict[str, str]]:
        agent.logger.log("📑 解析大纲章节...", level=LogLevel.DEBUG)
        agent.logger.log(f"大纲内容（前500字）:\n{outline[:500]}...", level=LogLevel.DEBUG)

        sections = []
        lines = outline.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match_numbered = re.match(r"^([\d.]+)\s+(.+)$", line)
            if match_numbered:
                raw_content = match_numbered.group(2).strip()
                number = match_numbered.group(1)
                OutlineParsingService.parse_and_add_section(agent, raw_content, sections, number=number)
                continue

            match_markdown = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match_markdown:
                raw_content = match_markdown.group(2).strip()
                level = len(match_markdown.group(1))
                OutlineParsingService.parse_and_add_section(agent, raw_content, sections, level=level)
                continue

        if not sections:
            agent.logger.log("⚠️ 标准格式解析失败，尝试容错方案...", level=LogLevel.INFO)
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("- ")
                    and not line.startswith("目标")
                    and "字" not in line
                    and not any(keyword in line for keyword in ["任务", "包含", "要求", "说明", "输出格式"])
                ):
                    title = re.sub(r"[\(\[\-].*", "", line).strip()

                    if title and len(title) > 2:
                        sections.append(
                            {
                                "title": title,
                                "goal": title,
                                "word_count_target": 0,
                                "fallback": True,
                            }
                        )

        if sections:
            OutlineParsingService.mark_leaf_sections(sections)

        agent.logger.log(f"✅ 解析完成，共 {len(sections)} 个章节", level=LogLevel.INFO)
        if sections:
            agent.logger.log("章节列表:", level=LogLevel.INFO)
            for i, sec in enumerate(sections[:10], 1):
                agent.logger.log(f"  {i}. {sec['title']}", level=LogLevel.INFO)
            if len(sections) > 10:
                agent.logger.log(f"  ... 还有 {len(sections) - 10} 个章节", level=LogLevel.INFO)
        else:
            agent.logger.log("⚠️ 未能解析出任何章节！", level=LogLevel.ERROR)
            agent.logger.log("问题诊断:", level=LogLevel.ERROR)
            agent.logger.log(f"  - 大纲总行数: {len(lines)}", level=LogLevel.ERROR)
            agent.logger.log(f"  - 非空行数: {len([l for l in lines if l.strip()])}", level=LogLevel.ERROR)
            agent.logger.log(f"  - 完整大纲内容:\n{outline}", level=LogLevel.ERROR)

        return sections

    @staticmethod
    def mark_leaf_sections(sections: List[Dict[str, Any]]) -> None:
        if not sections:
            return

        def normalize_number(raw: str) -> str:
            return str(raw or "").strip().rstrip(".")

        for index, section in enumerate(sections):
            is_leaf = True
            current_number = normalize_number(section.get("number", ""))
            current_level = section.get("level")

            if current_number:
                prefix = current_number + "."
                for other in sections[index + 1 :]:
                    other_number = normalize_number(other.get("number", ""))
                    if other_number and other_number.startswith(prefix):
                        is_leaf = False
                        break
            elif current_level is not None:
                for other in sections[index + 1 :]:
                    other_level = other.get("level")
                    if other_level is None:
                        continue
                    if other_level <= current_level:
                        break
                    if other_level > current_level:
                        is_leaf = False
                        break

            section["is_leaf"] = is_leaf
            if not is_leaf:
                section["word_count_target"] = 0

    @staticmethod
    def parse_and_add_section(agent, raw_content: str, sections: List, number: str = None, level: int = None):
        metadata = {}
        word_count_target = 0

        if " | " in raw_content:
            parts = [p.strip() for p in raw_content.split(" | ")]
            title = parts[0]

            for part in parts[1:]:
                if part.startswith("字数"):
                    match = re.search(r"\d+", part)
                    if match:
                        word_count_target = int(match.group())
                elif part.startswith("目标"):
                    metadata["goal"] = re.sub(r"^目标[：:]\s*", "", part)
                elif part.startswith("写作目标"):
                    metadata["writing_goal"] = re.sub(r"^写作目标[：:]\s*", "", part)
                elif part.startswith("逻辑关系"):
                    metadata["logic_relation"] = re.sub(r"^逻辑关系[：:]\s*", "", part)
                elif part.startswith("核心论点"):
                    metadata["core_argument"] = re.sub(r"^核心论点[：:]\s*", "", part)
                elif part.startswith("数据"):
                    metadata["data"] = re.sub(r"^数据[：:]\s*", "", part)
                elif part.startswith("问题"):
                    metadata["question"] = re.sub(r"^问题[：:]\s*", "", part)
                else:
                    if ":" in part:
                        key, val = part.split(":", 1)
                        metadata[key.strip().lower()] = val.strip()
        else:
            title = raw_content

        title, inline_word_count = OutlineParsingService.extract_title_and_word_count(title)
        if inline_word_count and word_count_target == 0:
            word_count_target = inline_word_count

        if title.startswith("核心论点：") or title.startswith("核心论点:"):
            metadata["core_argument"] = re.sub(r"^核心论点[：:]\s*", "", title)
            title = metadata["core_argument"]

        goal = metadata.get("writing_goal") or metadata.get("goal") or metadata.get("core_argument") or title

        section_dict = {
            "title": title,
            "goal": goal,
            "word_count_target": word_count_target,
            "metadata": metadata,
            "is_leaf": True,
        }

        if number is not None:
            section_dict["number"] = number
        if level is not None:
            section_dict["level"] = level

        sections.append(section_dict)

    @staticmethod
    def extract_title_and_word_count(title: str) -> Tuple[str, int]:
        if not title:
            return "", 0

        text = title.strip()
        match = re.search(r"\(约\s*(\d+)\s*字\)\s*$", text)
        if not match:
            return text, 0

        word_count = int(match.group(1))
        clean_title = re.sub(r"\(约\s*\d+\s*字\)\s*$", "", text).strip()
        return clean_title, word_count

    @staticmethod
    def sanitize_outline_for_writing(outline: str) -> str:
        if not outline:
            return ""

        cleaned_lines = []
        for line in str(outline).splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            sanitized = re.sub(r"\s*\|\s*字数[：:]\s*[^|\n]+", "", line)
            sanitized = re.sub(r"\s*\(约\s*\d+\s*字\)\s*", "", sanitized)
            cleaned_lines.append(sanitized.rstrip())

        return "\n".join(cleaned_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="OutlineParsingService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("OutlineParsingService static methods:")
        print("- parse_sections(agent, outline)")
        print("- mark_leaf_sections(sections)")
        print("- parse_and_add_section(agent, raw_content, sections, number=None, level=None)")
        print("- extract_title_and_word_count(title)")
        print("- sanitize_outline_for_writing(outline)")
        return 0

    print("这是 service 文件，不是可直接执行组件。")
    print("请调试组件文件，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_outline.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
