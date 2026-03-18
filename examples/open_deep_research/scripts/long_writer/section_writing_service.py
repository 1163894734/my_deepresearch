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

        输入参数：
        - authors (str): 作者列表的字符串表示，用于构造文内引用、canonical_key 或参考文献条目。
        - year (str): 年份字符串，用于检索约束、canonical_key 构造和文内引用匹配。
        - fallback_title (str): 当缺少作者和年份时用于降级构造键值的标题。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
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

        输入参数：
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
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

        输入参数：
        - section (Any): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - default_title (str): 当章节缺少标题时使用的默认标题。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
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

        输入参数：
        - raw (Any): 未经规范化的原始输入，可为字典、列表、字符串或混合结构。

        返回值：
        - Dict[str, Dict[str, str]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
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
    def _build_section_payload(payload: dict, section_type: str) -> Dict[str, Any]:
        """
        主要作用：按章节类型整理出实际写作所需输入。

        输入参数：
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """
        raw = SectionWritingService._unwrap_payload(payload)
        section_defaults = {
            "body": "正文",
            "introduction": "引言",
            "conclusion": "结论",
            "abstract": "摘要",
        }
        section = SectionWritingService._normalize_section(raw.get("section", {}), section_defaults.get(section_type, ""))
        available_citations = SectionWritingService._normalize_available_citations(raw.get("available_citations"))

        normalized: Dict[str, Any] = {
            "section": section,
            "task": str(raw.get("task", "") or ""),
            "available_citations": available_citations,
        }

        if section_type == "body":
            normalized["fine_rag_context"] = str(raw.get("fine_rag_context", "") or "")
            normalized["prev_section_content"] = str(raw.get("prev_section_content", "") or "")
        elif section_type == "introduction":
            normalized["prev_section_content"] = str(raw.get("prev_section_content", "") or "")
            normalized["fine_rag_context"] = str(raw.get("fine_rag_context", "") or "")
            normalized["full_text"] = str(raw.get("full_text", "") or "")
        elif section_type == "conclusion":
            normalized["full_text"] = str(raw.get("full_text", "") or "")
        elif section_type == "abstract":
            normalized["conclusion_text"] = str(raw.get("conclusion_text", "") or "")
            normalized["full_text"] = str(raw.get("full_text", "") or "")

        return normalized

    @staticmethod
    def write_intro_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成引言章节内容。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        import json
        normalized_payload = SectionWritingService._build_section_payload(payload, "introduction")
        section = normalized_payload.get("section", {})
        word_count_target = int(section.get("word_count_target", 0) or 0)
        prompt = (
            f"输入JSON：\n{json.dumps(normalized_payload, ensure_ascii=False, indent=2)}"
        )
        available_citations = normalized_payload.get("available_citations")
        try:
            content = agent.execute_tool_call("introduction_write", {"input": prompt})
        except Exception as e:
            agent.logger.log(f"⚠️ 直接json引言写作失败: {e}", level=LogLevel.ERROR)
            raise
        return str(SectionWritingService.introduction_reflection_loop(agent, content, section, word_count_target, available_citations))

    @staticmethod
    def write_conclusion_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成结论章节内容。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        import json
        normalized_payload = SectionWritingService._build_section_payload(payload, "conclusion")
        prompt = (
            f"输入JSON：\n{json.dumps(normalized_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            content = agent.execute_tool_call("conclusion_write", {"input": prompt})
        except Exception as e:
            agent.logger.log(f"⚠️ 直接json结论写作失败: {e}", level=LogLevel.ERROR)
            raise
        return str(content)

    @staticmethod
    def write_abstract_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成摘要章节内容。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        import json
        normalized_payload = SectionWritingService._build_section_payload(payload, "abstract")
        prompt = (
            f"输入JSON：\n{json.dumps(normalized_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            content = agent.execute_tool_call("abstract_write", {"input": prompt})
        except Exception as e:
            agent.logger.log(f"⚠️ 直接json摘要写作失败: {e}", level=LogLevel.ERROR)
            raise
        return str(content)

    @staticmethod
    def write_body_content_with_json(agent, payload: dict) -> str:
        """
        主要作用：通过 JSON 载荷生成正文章节内容。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        import json
        normalized_payload = SectionWritingService._build_section_payload(payload, "body")
        section = normalized_payload.get("section", {})
        word_count_target = int(section.get("word_count_target", 0) or 0)
        # 直接将payload序列化为json字符串作为prompt
        prompt = (
            f"输入JSON：\n{json.dumps(normalized_payload, ensure_ascii=False, indent=2)}"
        )
        available_citations = normalized_payload.get("available_citations")
        try:
            content = agent.execute_tool_call("section_write", {"input": prompt})
        except Exception as e:
            agent.logger.log(f"⚠️ 直接json写作失败: {e}", level=LogLevel.ERROR)
            raise
        return str(SectionWritingService.section_reflection_loop(agent, content, section, word_count_target, available_citations))

    """章节写作服务：细粒度检索、输入构造、正文与摘要写作、反思。"""

    @staticmethod
    def run_fine_rag_web_search(agent, section_title: str, search_query: str) -> str:
        """
        主要作用：执行细粒度网页检索。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - search_query (str): 针对某个章节生成的细粒度检索查询。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
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
    def build_section_input(
        agent,
        section: Dict[str, str],
        section_type: str,
        fine_rag_context: str = "",
        available_citations: Dict[str, Dict[str, str]] = None,
    ) -> str:
        """
        主要作用：构建正文章节写作输入提示词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。
        - fine_rag_context (str): 面向单个章节的细粒度检索上下文。
        - available_citations (Dict[str, Dict[str, str]]): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """

        task_text = agent._trim_text(agent._clean_task_description(), 600)
        outline_text = agent._trim_text(
            OutlineParsingService.sanitize_outline_for_writing(agent._current_outline),
            agent.max_outline_context_chars,
        )
        prev_text = agent._trim_text(agent._previous_section_content, agent.max_prev_section_chars)
        coarse_text = agent._trim_text(agent.state.get(agent.STATE_COARSE_RAG_CONTEXT, ""), 800)

        word_count_target = section.get("word_count_target", 0)
        word_count_hint = ""
        if word_count_target > 0:
            word_count_tolerance = int(word_count_target * 0.2)
            word_count_hint = (
                f"\n\n【字数要求】\n目标字数: {word_count_target}字（允许范围: "
                f"{word_count_target - word_count_tolerance}-{word_count_target + word_count_tolerance}字）\n"
                "请确保输出内容符合字数要求。"
            )

        available_cites_text = ""
        if available_citations:
            available_cites_text = "\n\n【严格引用白名单】\n"
            available_cites_text += format_allowed_citation_whitelist(available_citations) + "\n"
            available_cites_text += (
                "强约束：\n"
                "- 只能使用以上白名单中的括号式引用；\n"
                "- 不得输出任何未在白名单中的作者名和年份；\n"
                "- 不得使用叙述式引用（如 X et al. (2020)）；\n"
                "- 若当前句没有证据支撑，就不要加引用。"
            )
        elif agent.strict_citation_flow:
            available_cites_text = (
                "\n\n【严格引用白名单】\n"
                "当前段落没有可用引用。强约束：禁止输出任何 APA 文内引用，禁止编造作者与年份。"
            )

        if section_type == "references":
            bibliography = agent._trim_text(agent._format_references_section(), agent.max_bibliography_chars)
            return (
                f"用户prompt:\n{task_text}\n\n"
                f"大纲:\n{outline_text}\n\n"
                f"参考文献库:\n{bibliography}\n\n"
                f"当前章节: {section.get('title', '')}"
            )

        if section_type == "body":
            return (
                f"用户prompt:\n{task_text}\n\n"
                f"大纲:\n{outline_text}\n\n"
                f"web_search结果:\n{fine_rag_context or '（无）'}\n\n"
                f"上一段落内容:\n{prev_text or '（无）'}\n\n"
                f"当前章节: {section.get('title', '')}\n"
                f"章节目标: {section.get('goal', '')}{available_cites_text}{word_count_hint}"
            )

        if section_type == "introduction":
            intro_context = fine_rag_context or coarse_text
            return (
                f"用户prompt:\n{task_text}\n\n"
                f"大纲:\n{outline_text}\n\n"
                f"全文内容/背景材料:\n{intro_context or '（无）'}\n\n"
                f"当前章节: {section.get('title', '')}{available_cites_text}{word_count_hint}"
            )

        if section_type == "conclusion":
            conclusion_context = fine_rag_context or agent._trim_text(agent._global_summary, 1000)
            return (
                f"用户prompt:\n{task_text}\n\n"
                f"大纲:\n{outline_text}\n\n"
                f"全文内容/总结材料:\n{conclusion_context or '（无）'}\n\n"
                f"当前章节: {section.get('title', '')}{available_cites_text}{word_count_hint}"
            )

        return (
            f"用户prompt:\n{task_text}\n\n"
            f"大纲:\n{outline_text}\n\n"
            f"全文摘要:\n{agent._trim_text(agent._global_summary, 600) or '（无）'}\n\n"
            f"当前章节: {section.get('title', '')}{available_cites_text}{word_count_hint}"
        )

    @staticmethod
    def build_abstract_input(agent, section: Dict[str, str], conclusion_text: str) -> str:
        """
        主要作用：构建摘要写作输入提示词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - conclusion_text (str): 结论章节正文，通常作为摘要生成的输入材料。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """

        task_text = agent._trim_text(agent._clean_task_description(), 600)
        outline_text = agent._trim_text(
            OutlineParsingService.sanitize_outline_for_writing(agent._current_outline),
            agent.max_outline_context_chars,
        )
        summary_text = agent._trim_text(agent._global_summary, 1000) or "（无）"
        conclusion_text = agent._trim_text(conclusion_text, 900) or "（无）"

        word_count_target = section.get("word_count_target", 0)
        word_count_hint = ""
        if word_count_target > 0:
            tolerance = int(word_count_target * 0.2)
            word_count_hint = (
                f"\n\n【字数要求】\n"
                f"目标字数: {word_count_target}字\n"
                f"允许范围: {word_count_target - tolerance}-{word_count_target + tolerance}字"
            )

        return (
            f"用户prompt:\n{task_text}\n\n"
            f"大纲（写作阶段已隐藏字数）:\n{outline_text}\n\n"
            f"全文关键内容摘要:\n{summary_text}\n\n"
            f"结论章节草稿:\n{conclusion_text}\n\n"
            f"当前章节: {section.get('title', '摘要')}\n"
            f"章节目标: {section.get('goal', '凝练全文核心发现、方法与结论，形成可独立阅读的摘要')}{word_count_hint}\n\n"
            f"写作要求:\n"
            f"1. 摘要必须可独立阅读，覆盖研究背景、核心问题、主要发现与结论价值。\n"
            f"2. 必须严格基于正文与结论已出现的信息，禁止引入新事实。\n"
            f"3. 语言紧凑、客观、学术化，不写空泛套话。"
        )

    @staticmethod
    def write_abstract_section(agent, section: Dict[str, str], conclusion_text: str) -> str:
        """
        主要作用：直接调用摘要技能生成摘要文本。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - conclusion_text (str): 结论章节正文，通常作为摘要生成的输入材料。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        try:
            abstract_input = SectionWritingService.build_abstract_input(agent, section, conclusion_text)
            content = agent.execute_tool_call("abstract_write", {"input": abstract_input})
            return str(content)
        except Exception as e:
            agent.logger.log(f"⚠️ 摘要生成失败: {e}", level=LogLevel.ERROR)
            raise

    @staticmethod
    def build_skeleton_planning_input(agent, section: Dict[str, str], fact_list: str) -> str:
        """
        主要作用：构建章节骨架规划提示词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - fact_list (str): 结构化或半结构化的事实材料列表文本。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """

        outline_text = agent._trim_text(
            OutlineParsingService.sanitize_outline_for_writing(agent._current_outline),
            500,
        )
        prev_summary = agent._trim_text(agent._previous_section_content[:300], 200)

        return (
            f"段落标题：{section.get('title', '')}\n"
            f"位置：{section.get('position', 'body')}\n"
            f"前文关键要点：{prev_summary or '（无）'}\n\n"
            f"大纲（全局上下文）：\n{outline_text}\n\n"
            f"事实清单：\n{fact_list}\n\n"
            f"请基于事实清单，设计 3-5 句的段落逻辑骨架，\n"
            f"遵循 Answer First 原则（第一句给出核心论点），\n"
            f"明确每句使用的核心数据和论证关系。"
        )

    @staticmethod
    def build_composition_input(agent, section: Dict[str, str], fact_list: str, skeleton: str, word_count_target: int = 0) -> str:
        """
        主要作用：构建章节成稿提示词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - fact_list (str): 结构化或半结构化的事实材料列表文本。
        - skeleton (str): 章节骨架计划或提纲式草稿。
        - word_count_target (int): 章节目标字数，用于控制生成长度。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """

        word_count_hint = ""
        if word_count_target > 0:
            word_count_tolerance = int(word_count_target * 0.2)
            word_count_hint = (
                f"\n\n【字数要求】\n"
                f"目标字数: {word_count_target}字\n"
                f"允许范围: {word_count_target - word_count_tolerance}-{word_count_target + word_count_tolerance}字"
            )

        citation_hint = ""
        available_citations = section.get("_available_citations") if isinstance(section, dict) else None
        if isinstance(available_citations, dict) and available_citations:
            citation_hint = (
                "\n\n【严格引用白名单】\n"
                f"{format_allowed_citation_whitelist(available_citations)}\n"
                "只能使用以上括号式引用；不要使用任何白名单之外的作者与年份。"
            )
        elif agent.strict_citation_flow:
            citation_hint = "\n\n【严格引用白名单】\n当前段落没有可用引用，禁止输出任何 APA 文内引用。"

        return (
            f"段落标题：{section.get('title', '')}\n"
            f"风格要求：冷静、笃定、权威、数据驱动（麦肯锡风格）{word_count_hint}{citation_hint}\n\n"
            f"逻辑骨架：\n{skeleton}\n\n"
            f"事实清单：\n{fact_list}\n\n"
            f"请严格按照骨架的逻辑顺序，用顶级咨询公司的语言风格，\n"
            f"将数据融合为完整、流畅的段落文本。\n"
            f"确保：\n"
            f"- 遵循骨架的 3-5 句逻辑顺序\n"
            f"- 只有在白名单存在可用来源时，关键数据才可加 APA 文内标注\n"
            f"- 避免第一人称和模糊表述\n"
            f"- 使用冷静、权威的表达方式"
        )

    @staticmethod
    def write_body_content(
        agent,
        section: Dict[str, Any],
        fine_rag_context: str,
        available_citations: Dict[str, Dict[str, str]],
    ) -> str:
        """
        主要作用：执行正文章节的骨架规划与成稿。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - section (Dict[str, Any]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - fine_rag_context (str): 面向单个章节的细粒度检索上下文。
        - available_citations (Dict[str, Dict[str, str]]): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
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

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - word_count_target (int): 章节目标字数，用于控制生成长度。
        - available_citations (Dict): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
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

    @staticmethod
    def introduction_reflection_loop(
        agent,
        text: str,
        section: Dict[str, str],
        word_count_target: int = 0,
        available_citations: Dict = None,
    ) -> str:
        """
        主要作用：执行引言章节的反思—修订循环。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。
        - word_count_target (int): 章节目标字数，用于控制生成长度。
        - available_citations (Dict): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        current = text
        section_title = section.get("title", "引言")
        goal = section.get("goal", "")

        try:
            # 1. 反思：调用 evidence_reflection skill
            agent.logger.log("  🔍 引言反思...", level=LogLevel.DEBUG)
            reflection_input = (
                f"章节标题：{section_title}\n"
                f"章节目标：{goal}\n\n"
                f"内容：\n{current}"
            )
            report = str(agent.execute_tool_call("evidence_reflection", {"input": reflection_input}))
            agent._log_reflection_to_file(f"引言反思 - {goal[:30]}", 0, report)

            # 2. 修订：将原文 + 反思报告交给 section_revision skill
            revision_prompt = (
                f"章节标题：{section_title}\n\n"
                f"原始内容：\n{current}\n\n"
                f"反思报告：\n{report}\n\n"
                "请根据反思报告修订引言，直接输出修订后的完整引言，不要输出任何解释。"
            )
            current = str(agent.execute_tool_call("section_revision", {"input": revision_prompt}))
            agent._log_revision_to_file(f"引言修订 - {goal[:30]}", current)
            agent.logger.log("  ✅ 引言修订完成", level=LogLevel.DEBUG)

        except Exception as e:
            agent.logger.log(f"  ⚠️ 引言反思出错: {e}", level=LogLevel.ERROR)
            raise

        return current


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
