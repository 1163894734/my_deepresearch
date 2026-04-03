import argparse
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

class OutlineParsingService:
    """大纲解析服务：章节抽取、叶子节点标注与写作前清洗。（已解除对 Agent 的依赖，作为纯函数库使用）"""

    @staticmethod
    def parse_sections(
        outline: str, 
        log_callable: Optional[Callable[[str, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        主要作用：把大纲文本解析为章节结构列表。
        
        参数：
        - outline (str): 待解析的大纲文本。
        - log_callable (Callable): 可选的日志回调函数，格式为 func(msg: str, level: str)
        """
        
        # 内部日志包装器
        def log(msg: str, level: str = "INFO"):
            if log_callable:
                log_callable(msg, level)

        log("📑 解析大纲章节...", "DEBUG")
        log(f"大纲内容（前500字）:\n{outline[:500]}...", "DEBUG")

        sections = []
        lines = outline.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 匹配 "1.1.2 章节标题" 格式
            match_numbered = re.match(r"^([\d.]+)\s+(.+)$", line)
            if match_numbered:
                raw_content = match_numbered.group(2).strip()
                number = match_numbered.group(1)
                OutlineParsingService.parse_and_add_section(raw_content, sections, number=number)
                continue

            # 匹配 "### 章节标题" Markdown 格式
            match_markdown = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match_markdown:
                raw_content = match_markdown.group(2).strip()
                level = len(match_markdown.group(1))
                OutlineParsingService.parse_and_add_section(raw_content, sections, level=level)
                continue

        # 容错降级方案
        if not sections:
            log("⚠️ 标准格式解析失败，尝试容错方案...", "INFO")
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

        log(f"✅ 解析完成，共 {len(sections)} 个章节", "INFO")
        if sections:
            log("章节列表:", "INFO")
            for i, sec in enumerate(sections[:10], 1):
                log(f"  {i}. {sec['title']}", "INFO")
            if len(sections) > 10:
                log(f"  ... 还有 {len(sections) - 10} 个章节", "INFO")
        else:
            log("⚠️ 未能解析出任何章节！", "ERROR")
            log("问题诊断:", "ERROR")
            log(f"  - 大纲总行数: {len(lines)}", "ERROR")
            log(f"  - 非空行数: {len([l for l in lines if l.strip()])}", "ERROR")
            log(f"  - 完整大纲内容:\n{outline}", "ERROR")

        return sections

    @staticmethod
    def mark_leaf_sections(sections: List[Dict[str, Any]]) -> None:
        """为章节树中的叶子节点打标"""
        if not sections:
            return

        def normalize_number(raw: str) -> str:
            return str(raw or "").strip().rstrip(".")

        for index, section in enumerate(sections):
            is_leaf = True
            current_number = normalize_number(section.get("number", ""))
            current_level = section.get("level")

            # 通过数字编号判断
            if current_number:
                prefix = current_number + "."
                for other in sections[index + 1 :]:
                    other_number = normalize_number(other.get("number", ""))
                    if other_number and other_number.startswith(prefix):
                        is_leaf = False
                        break
            # 通过 Markdown 层级判断
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
            # 非叶子节点（即父目录）不设定写作字数目标
            if not is_leaf:
                section["word_count_target"] = 0

    @staticmethod
    def parse_and_add_section(raw_content: str, sections: List, number: str = None, level: int = None):
        """解析单段大纲文本（如含字数、目标等 metadata）并追加到章节列表"""
        metadata = {}
        word_count_target = 0

        # 解析管道符分隔的元数据 "标题 | 字数: 500 | 目标: xxx"
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

        # 解析括号内的字数 "(约 500 字)"
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
        """从标题中提取纯标题与目标字数"""
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
        """清洗大纲文本，去掉写作阶段不需要的元数据噪声"""
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
    parser = argparse.ArgumentParser(description="OutlineParsingService 纯净版调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("OutlineParsingService static methods:")
        print("- parse_sections(outline, log_callable=None)")
        print("- mark_leaf_sections(sections)")
        print("- parse_and_add_section(raw_content, sections, number=None, level=None)")
        print("- extract_title_and_word_count(title)")
        print("- sanitize_outline_for_writing(outline)")
        return 0

    print("✅ 这是纯净的 service 文件，已完全解除对 Agent 框架的依赖，可作为独立库函数使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())