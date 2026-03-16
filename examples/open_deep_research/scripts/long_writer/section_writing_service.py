from __future__ import annotations

import argparse
import re
from typing import Any, Dict

from smolagents.monitoring import LogLevel

try:
    from .outline_parsing_service import OutlineParsingService
except ImportError:
    from outline_parsing_service import OutlineParsingService


class SectionWritingService:
    """章节写作服务：细粒度检索、输入构造、正文与摘要写作、反思。"""

    @staticmethod
    def run_fine_rag_web_search(agent, section_title: str, search_query: str) -> str:
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

        agent.logger.log(f"⚠️ web_search不可用，章节【{section_title}】将基于已有上下文生成", level=LogLevel.INFO)
        return ""

    @staticmethod
    def build_section_input(
        agent,
        section: Dict[str, str],
        section_type: str,
        fine_rag_context: str = "",
        available_citations: Dict[str, Dict[str, str]] = None,
    ) -> str:
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
            available_cites_text += agent._citation_flow_service.format_allowed_citation_whitelist(agent, available_citations) + "\n"
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
        try:
            abstract_input = SectionWritingService.build_abstract_input(agent, section, conclusion_text)
            content = agent.execute_tool_call("abstract_write", {"input": abstract_input})
            return str(content)
        except Exception as e:
            agent.logger.log(f"⚠️ 摘要生成失败: {e}", level=LogLevel.ERROR)
            return ""

    @staticmethod
    def build_skeleton_planning_input(agent, section: Dict[str, str], fact_list: str) -> str:
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
                f"{agent._citation_flow_service.format_allowed_citation_whitelist(agent, available_citations)}\n"
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
        word_count_target = int(section.get("word_count_target", 0) or 0)

        skeleton_input = SectionWritingService.build_skeleton_planning_input(agent, section, fine_rag_context)
        try:
            skeleton = agent.execute_tool_call("section_skeleton_planning", {"input": skeleton_input})
            agent.logger.log("✅ 骨架规划完成", level=LogLevel.DEBUG)
        except Exception as e:
            agent.logger.log(f"⚠️ 骨架规划失败: {e}，降级使用原始 section_write", level=LogLevel.INFO)
            body_input = SectionWritingService.build_section_input(agent, section, "body", fine_rag_context, available_citations)
            content = agent.execute_tool_call("section_write", {"input": body_input})
            return str(SectionWritingService.section_reflection_loop(agent, content, section, word_count_target))

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
            agent.logger.log(f"⚠️ 文本组装失败: {e}，降级为骨架展开", level=LogLevel.INFO)
            content = skeleton

        return str(SectionWritingService.section_reflection_loop(agent, content, section, word_count_target))

    @staticmethod
    def section_reflection_loop(agent, text: str, section: Dict[str, str], word_count_target: int = 0) -> str:
        current = text
        current_words = len(current)
        section_title = section.get("title", "未命名章节")
        goal = section.get("goal", "")

        try:
            if word_count_target > 0:
                word_count_tolerance = int(word_count_target * 0.2)
                min_words = word_count_target - word_count_tolerance
                max_words = word_count_target + word_count_tolerance

                agent.logger.log(
                    f"  📏 字数检查: {current_words}字 (目标: {word_count_target}±{word_count_tolerance}字)",
                    level=LogLevel.DEBUG,
                )

                if current_words < min_words or current_words > max_words:
                    agent.logger.log("  ⚠️ 字数不达标，先调整字数...", level=LogLevel.INFO)

                    word_adjust_prompt = f"""【内容】
{current}

【字数调整要求】
当前字数: {current_words}字
目标字数: {word_count_target}字（允许范围: {min_words}-{max_words}字）

【任务】
"""
                    if current_words < min_words:
                        word_adjust_prompt += f"""当前字数不足，需要扩充内容：
1. 补充更多论证和例子
2. 展开关键论点的细节
3. 增加过渡和解释性文字
4. 确保内容达到 {min_words}-{max_words} 字范围"""
                    else:
                        word_adjust_prompt += f"""当前字数超标，需要精简内容：
1. 删除冗余表述和重复内容
2. 合并相似论点
3. 保留核心观点和关键证据
4. 确保内容在 {min_words}-{max_words} 字范围"""

                    word_adjust_prompt += """

【输出要求】
直接输出调整后的完整段落，不要有任何标记或说明，只输出最终文本。"""

                    current = str(agent.execute_tool_call("section_revision", {"input": word_adjust_prompt}))

                    adjusted_words = len(current)
                    agent.logger.log(f"  📝 字数调整完成: {current_words}字 → {adjusted_words}字", level=LogLevel.INFO)

                    agent._log_revision_to_file(f"字数调整 - {goal[:30]}...", current)

                    if adjusted_words < min_words or adjusted_words > max_words:
                        agent.logger.log(f"  ⚠️ 字数仍未达标: {adjusted_words}字", level=LogLevel.INFO)
                    else:
                        agent.logger.log("  ✅ 字数已达标", level=LogLevel.DEBUG)

                    current_words = adjusted_words
                else:
                    agent.logger.log("  ✅ 字数合格", level=LogLevel.DEBUG)

            agent.logger.log("  🔍 证据反思...", level=LogLevel.DEBUG)

            clean_goal = goal.split("|")[0].strip() if "|" in goal else goal
            clean_goal = re.sub(r"[：:].*$", "", clean_goal).strip()

            reflection_input = (
                f"章节标题\n\n{section_title}\n\n"
                f"章节目标\n\n{clean_goal}\n\n"
                f"内容\n\n{current}"
            )

            report = str(agent.execute_tool_call("evidence_reflection", {"input": reflection_input}))
            data = agent._parse_json(report)
            score = data.get("score", 0)
            evidence_satisfied = data.get("is_pass", False) or (score > 80)

            agent._log_reflection_to_file(f"证据反思 - {goal[:30]}...", score, report)

            if evidence_satisfied and score >= agent.section_score_threshold:
                agent.logger.log(f"  ✅ 得分 {score}，证据充分", level=LogLevel.DEBUG)
                return current

            agent.logger.log(f"  ⚠️ 得分 {score}，证据不足，进行修订...", level=LogLevel.INFO)

            evidence_revision_prompt = f"""【章节标题】
{section_title}

【内容】
{current}

【评审报告】
{report}

【修改要求】
1. 根据评审报告补充证据，增强论证的说服力
2. 保持当前的内容长度（{current_words}字左右）
3. 不要使用修改标记，直接输出最终文本

请输出修改后的完整段落。"""

            current = str(agent.execute_tool_call("section_revision", {"input": evidence_revision_prompt}))

            final_words = len(current)
            agent.logger.log(f"  ✅ 证据修订完成: {final_words}字", level=LogLevel.DEBUG)
            agent._log_revision_to_file(f"证据修订 - {goal[:30]}...", current)

        except Exception as e:
            agent.logger.log(f"  ⚠️ 反思出错: {e}", level=LogLevel.INFO)

        return current

    @staticmethod
    def introduction_reflection_loop(agent, text: str, section: Dict[str, str], word_count_target: int = 0) -> str:
        """引言专用后处理：仅做字数对齐，不调用证据反思机制。"""
        current = text
        current_words = len(current)
        section_title = section.get("title", "引言")
        goal = section.get("goal", "")

        try:
            if word_count_target > 0:
                tolerance = int(word_count_target * 0.2)
                min_words = word_count_target - tolerance
                max_words = word_count_target + tolerance

                agent.logger.log(
                    f"  📏 引言字数检查: {current_words}字 (目标: {word_count_target}±{tolerance}字)",
                    level=LogLevel.DEBUG,
                )

                if current_words < min_words or current_words > max_words:
                    adjust_prompt = f"""【章节类型】
引言

【当前内容】
{current}

【字数要求】
目标字数: {word_count_target}字
允许范围: {min_words}-{max_words}字

【任务】
请在不改变核心论点的前提下调整引言长度，使其落入目标范围。
直接输出调整后的完整引言，不要输出解释说明。"""

                    current = str(agent.execute_tool_call("section_revision", {"input": adjust_prompt}))
                    current_words = len(current)
                    agent._log_revision_to_file(f"引言字数调整 - {goal[:30]}...", current)

            agent.logger.log("  ℹ️ 引言已跳过证据反思，仅执行字数对齐", level=LogLevel.DEBUG)

        except Exception as e:
            agent.logger.log(f"  ⚠️ 引言反思出错: {e}", level=LogLevel.INFO)

        return current


def main() -> int:
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
