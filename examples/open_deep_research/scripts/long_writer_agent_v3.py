from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple
import json
import os
import re
import time

from smolagents import CustomAgent
from smolagents.agents import ActionOutput, ToolOutput
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel


@dataclass
class LongWriterRuntimeState:
    task: str = ""
    cleaned_task: str = ""
    outline: str = ""
    available_citations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    written_sections: List[Dict[str, Any]] = field(default_factory=list)
    current_section: Dict[str, Any] = field(default_factory=dict)
    current_section_content: str = ""
    previous_section_content: str = ""
    full_content: str = ""
    abstract_text: str = ""
    references_text: str = ""
    final_markdown: str = ""


class ReportWorkspace:
    """File and state manager for long report generation."""

    def __init__(self, base_dir: str, agent_logger):
        self.logger = agent_logger
        self.output_dir = base_dir
        self.output_file = os.path.join(self.output_dir, "report.md")
        self.final_output_file = os.path.join(self.output_dir, "report_final.md")
        self.log_file = os.path.join(self.output_dir, "generation_log.txt")
        self.reference_trace_file = os.path.join(self.output_dir, "reference_trace_log.txt")
        self.citations_validation_log = os.path.join(self.output_dir, "citations_validation.json")
        self.tagged_search_log_file = os.path.join(self.output_dir, "tagged_search_log.jsonl")
        self.max_skill_log_chars = 1200
        self._init_directories()

    def _init_directories(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.safe_write(self.output_file, f"# 生成的长文本报告\n\n初始化时间 {timestamp}\n\n---\n\n", "w")
        self.safe_write(self.reference_trace_file, f"# 参考文献追踪日志\n\n生成时间 {timestamp}\n\n---\n\n", "w")
        self.safe_write(self.tagged_search_log_file, f"# tagged web 检索结构化日志(JSONL)\n# 生成时间 {timestamp}\n", "w")

    def safe_write(self, file_path: str, content: str, mode: str = "a") -> None:
        try:
            with open(file_path, mode, encoding="utf-8") as file_handle:
                file_handle.write(content)
        except Exception as exc:
            self.logger.log(f"文件操作失败 {file_path} {exc}", level=LogLevel.ERROR)

    def get_heading_prefix(self, section_number: str = "", section_level: Optional[int] = None) -> str:
        if not section_number:
            return "##" if section_level is None or section_level <= 1 else ("###" if section_level == 2 else "####")
        normalized = str(section_number).strip().rstrip(".")
        if not normalized:
            return "##" if section_level is None or section_level <= 1 else ("###" if section_level == 2 else "####")
        depth = normalized.count(".") + 1
        return "##" if depth <= 1 else ("###" if depth == 2 else "####")

    def filter_references(self, content: str) -> str:
        pattern = r"\n(#+\s*(参考文献|References|引用|Bibliography)|\*\*(参考文献|References)\*\*).*"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return content[: match.start()].strip()
        return content.strip()

    def append_section(
        self,
        title: str,
        content: str,
        section_index: int = 0,
        total_sections: int = 0,
        section_type: str = "body",
        section_number: str = "",
        section_level: Optional[int] = None,
    ) -> None:
        progress = f" [{section_index}/{total_sections}]" if section_index > 0 and total_sections > 0 else ""
        heading_prefix = self.get_heading_prefix(section_number, section_level)
        numbered_title = f"{section_number} {title}".strip() if section_number else title
        filtered_content = content if section_type == "references" else self.filter_references(content)
        block = f"{heading_prefix} {numbered_title}{progress}\n\n{filtered_content}\n\n"
        self.safe_write(self.output_file, block)
        self.logger.log(f"✅ 段落已保存到文件{progress} {self.output_file}", level=LogLevel.INFO)

    def save_outline(self, outline: str) -> None:
        content = f"## 📋 文章大纲\n\n{outline}\n\n---\n\n"
        self.safe_write(self.output_file, content, "a")

    def write_final_report(self, markdown_content: str) -> None:
        self.safe_write(self.final_output_file, markdown_content, "w")

    def log_skill_call(
        self,
        tool_name: str,
        arguments: Any,
        output: Any = None,
        status: str = "success",
        elapsed_ms: Optional[int] = None,
        error_message: str = "",
        current_step: str = "unknown",
        resolved_skill_key: str = "",
    ) -> None:
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            full_input = json.dumps(arguments, ensure_ascii=False, indent=2) if isinstance(arguments, (dict, list)) else str(arguments)
            full_output = json.dumps(output, ensure_ascii=False, indent=2) if isinstance(output, (dict, list)) else (str(output) if output is not None else "")
            log_block = [
                f"\n## 🧩 [Skill调用] {tool_name} @ {timestamp}",
                f"- **状态**: `{'成功' if status == 'success' else '失败'}`",
                f"- **耗时**: `{elapsed_ms} ms`",
                f"- **当前阶段**: `{current_step}`",
                f"- **提示词映射**: `{resolved_skill_key or '未命中'}`",
                "\n### 输入参数 (Input)",
                full_input,
                "\n### 返回结果 (Output)" if status == "success" else "\n### 错误信息 (Error)",
                full_output if status == "success" else error_message,
                "\n---",
            ]
            self.safe_write(self.log_file, "\n".join(log_block))
        except Exception as exc:
            self.logger.log(f"⚠️ [系统异常] Skill日志写入失败: {exc}", level=LogLevel.ERROR)

    def log_reference_event(self, stage: str, section_title: str, detail: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.safe_write(self.reference_trace_file, f"[{timestamp}] [{stage}] 章节 {section_title or '未知章节'}\n{detail}\n\n")

    def append_validation_logs(self, step_logs: list) -> None:
        if not step_logs:
            return
        try:
            existing_data: Dict[str, Any] = {}
            if os.path.exists(self.citations_validation_log):
                with open(self.citations_validation_log, "r", encoding="utf-8") as file_handle:
                    try:
                        existing_data = json.load(file_handle)
                    except Exception:
                        existing_data = {}
            logs = existing_data.get("five_step_logs", [])
            if not isinstance(logs, list):
                logs = []
            logs.extend(step_logs)
            existing_data["five_step_logs"] = logs
            with open(self.citations_validation_log, "w", encoding="utf-8") as file_handle:
                json.dump(existing_data, file_handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.log(f"⚠️ 验证日志写入失败: {exc}", level=LogLevel.ERROR)


class LongWriterAgent(CustomAgent):
    STATE_OUTLINE = "outline"
    STATE_SECTIONS = "sections"

    def __init__(
        self,
        model,
        tools: Optional[List] = None,
        output_dir: Optional[str] = None,
        planning_agent=None,
        section_agent=None,
        finalization_agent=None,
        citation_validation_tool=None,
        writer_state: Optional[LongWriterRuntimeState] = None,
        **kwargs,
    ):
        super().__init__(model=model, tools=tools or [], **kwargs)

        if not hasattr(self.logger, "info"):
            self.logger.info = lambda msg, *args, **kwargs: self.logger.log(msg, level=LogLevel.INFO)
        if not hasattr(self.logger, "error"):
            self.logger.error = lambda msg, *args, **kwargs: self.logger.log(msg, level=LogLevel.ERROR)
        if not hasattr(self.logger, "warning"):
            self.logger.warning = lambda msg, *args, **kwargs: self.logger.log(msg, level=LogLevel.WARNING)

        if not output_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_project_dir, "outputs", f"report_Frontier_Review_{timestamp}")

        self.workspace = ReportWorkspace(output_dir, self.logger)
        self.planning_agent = planning_agent
        self.section_agent = section_agent
        self.finalization_agent = finalization_agent
        self.citation_validation_tool = citation_validation_tool
        self.runtime_state = writer_state or LongWriterRuntimeState()

        self._output_dir = self.workspace.output_dir
        self._output_file = self.workspace.output_file
        self._final_output_file = self.workspace.final_output_file
        self._log_file = self.workspace.log_file
        self._reference_trace_file = self.workspace.reference_trace_file
        self._citations_validation_log = self.workspace.citations_validation_log
        self._tagged_search_log_file = self.workspace.tagged_search_log_file

        self._current_outline = self.runtime_state.outline
        self._generated_sections: List[Dict[str, Any]] = self.runtime_state.written_sections
        self._abstract_section: Optional[Dict[str, Any]] = None
        self._previous_section_content = self.runtime_state.previous_section_content
        self._prev_body_or_intro_content = self.runtime_state.previous_section_content
        self._global_summary = ""
        self._full_text_body = self.runtime_state.full_content
        self._citations: Dict[str, Dict[str, Any]] = self.runtime_state.available_citations
        self._current_step = "init"
        self.max_summary_length = 300
        self.outline_max_iter = 2
        self.section_max_iter = 1
        self.outline_score_threshold = 90
        self.section_score_threshold = 85
        self.enable_citation_validation = True

        self.state["writer_state"] = self.runtime_state
        self.state["run_dir"] = self._output_dir
        self.state["current_phase"] = "init"
        self.state[self.STATE_OUTLINE] = self._current_outline
        self.state[self.STATE_SECTIONS] = self._generated_sections

        for child_agent in [self.planning_agent, self.section_agent, self.finalization_agent]:
            if child_agent is not None and hasattr(child_agent, "state"):
                child_agent.state["writer_state"] = self.runtime_state
                child_agent.state["run_dir"] = self._output_dir

    def set_text_webbrowser_agent_name(self, agent_name: str) -> None:
        self.text_webbrowser_agent_name = agent_name
        self.logger.log(f"✅ text_webbrowser_agent 已配置 {agent_name}", level=LogLevel.INFO)

    def enable_citation_validation_mode(self, enable: bool = True) -> None:
        self.enable_citation_validation = enable
        if enable:
            self.logger.log("✅ 启用学术引用验证系统", level=LogLevel.INFO)

    def _clean_task_description(self) -> str:
        task = str(self.task or "")
        task_marker = "Task:\n"
        if task_marker in task:
            task_content = task[task.find(task_marker) + len(task_marker) :]
            if "\n---\n" in task_content:
                task_content = task_content.split("\n---\n")[0]
            return task_content.strip()
        return task.strip()

    def _normalize_agent_payload(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            text = result.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return {"content": result}
        return {"content": str(result)}

    def _parse_outline_sections(self, outline: str) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*(.*?)\s*\|\s*字数：([0-9]+)字")
        for raw_line in str(outline or "").splitlines():
            line = raw_line.strip()
            if not line or "字数：" not in line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            word_count = int(match.group(3))
            if word_count <= 0:
                continue
            number = match.group(1).strip()
            title = match.group(2).strip()
            sections.append(
                {
                    "number": number,
                    "title": title,
                    "word_count": word_count,
                    "level": number.count(".") + 1,
                    "raw_line": line,
                }
            )
        return sections

    def _section_ref(self, section: Dict[str, Any]) -> str:
        number = str(section.get("number", "")).strip()
        title = str(section.get("title", "未命名章节")).strip() or "未命名章节"
        return f"{number} {title}".strip() if number else title

    def _run_agent(self, agent, prompt: str) -> Dict[str, Any]:
        result = agent.run(prompt)
        return self._normalize_agent_payload(result)

    def _planning_phase(self, memory_step: ActionStep) -> None:
        if self.planning_agent is None:
            raise ValueError("planning_agent 未注入，无法执行规划阶段")

        self.state["current_phase"] = "planning"
        cleaned_task = self._clean_task_description()
        self.runtime_state.task = self.task
        self.runtime_state.cleaned_task = cleaned_task

        prompt = json.dumps(
            {
                "task": cleaned_task,
                "run_dir": self._output_dir,
                "instruction": "先调用 get_planning_sop，然后返回 outline 和 available_citations。",
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._run_agent(self.planning_agent, prompt)
        outline = str(payload.get("outline") or payload.get("content") or "").strip()
        citations = payload.get("available_citations") or payload.get("citations") or {}
        if not isinstance(citations, dict):
            citations = {}

        self._current_outline = outline
        self._citations.clear()
        self._citations.update(citations)
        self.runtime_state.outline = outline
        self.runtime_state.available_citations = self._citations
        self.workspace.save_outline(outline)

    def _writing_phase(self, memory_step: ActionStep) -> None:
        if self.section_agent is None:
            raise ValueError("section_agent 未注入，无法执行章节写作阶段")

        self.state["current_phase"] = "writing"
        sections = self._parse_outline_sections(self._current_outline)
        self.runtime_state.sections = sections
        previous_section_content = self._previous_section_content
        total_sections = len(sections)

        if not sections:
            raise ValueError("未能从 outline 中解析出任何章节，请先修复规划阶段输出")

        self.section_agent.state["writer_state"] = self.runtime_state
        self.section_agent.state["current_phase"] = "writing"

        written_sections: List[Dict[str, Any]] = []
        rendered_sections: List[str] = []

        for index, section in enumerate(sections, start=1):
            section_ref = self._section_ref(section)
            self.runtime_state.current_section = section
            self.runtime_state.previous_section_content = previous_section_content

            prompt = json.dumps(
                {
                    "task": self.runtime_state.cleaned_task,
                    "outline": self._current_outline,
                    "current_section": section,
                    "current_section_ref": section_ref,
                    "previous_section_content": previous_section_content,
                    "available_citations": list(self._citations.keys()),
                    "section_index": index,
                    "section_total": total_sections,
                },
                ensure_ascii=False,
                indent=2,
            )
            payload = self._run_agent(self.section_agent, prompt)
            section_content = str(payload.get("content") or payload.get("section_content") or "").strip()
            citations_used = payload.get("citations_used") or payload.get("citations") or []
            final_score = payload.get("final_score", payload.get("score", 0))

            if self.citation_validation_tool is not None:
                validation_raw = self.citation_validation_tool.forward(
                    citations=self._citations,
                    paragraph=section_content,
                    section_ref=section_ref,
                )
                validation_payload = self._normalize_agent_payload(validation_raw)
                updated_citations = validation_payload.get("available_citations", {})
                if isinstance(updated_citations, dict) and updated_citations:
                    self._citations.clear()
                    self._citations.update(updated_citations)
                section_content = str(validation_payload.get("paragraph", section_content)).strip()

            section_record = {
                "title": section.get("title", "未命名章节"),
                "type": "body",
                "number": section.get("number", ""),
                "level": section.get("level"),
                "content": section_content,
                "citations": citations_used,
                "final_score": final_score,
            }
            written_sections.append(section_record)
            rendered_sections.append(f"## {section_ref}\n\n{section_content}".strip())
            self.workspace.append_section(
                section_record["title"],
                section_content,
                section_index=index,
                total_sections=total_sections,
                section_type=section_record["type"],
                section_number=str(section_record["number"]),
                section_level=section_record["level"],
            )
            previous_section_content = section_content

        self._generated_sections = written_sections
        self._full_text_body = "\n\n".join(rendered_sections).strip()
        self._previous_section_content = previous_section_content
        self.runtime_state.written_sections = written_sections
        self.runtime_state.full_content = self._full_text_body
        self.runtime_state.available_citations = self._citations
        self.runtime_state.previous_section_content = previous_section_content

    def _finalization_phase(self, memory_step: ActionStep) -> None:
        if self.finalization_agent is None:
            raise ValueError("finalization_agent 未注入，无法执行整合阶段")

        self.state["current_phase"] = "finalization"
        self.runtime_state.full_content = self._full_text_body
        self.runtime_state.written_sections = self._generated_sections
        self.runtime_state.available_citations = self._citations

        self.finalization_agent.state["writer_state"] = self.runtime_state
        self.finalization_agent.state["current_phase"] = "finalization"

        prompt = json.dumps(
            {
                "task": self.runtime_state.cleaned_task,
                "outline": self._current_outline,
                "full_content": self._full_text_body,
                "written_sections": self._generated_sections,
                "available_citations": self._citations,
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._run_agent(self.finalization_agent, prompt)
        abstract_text = str(payload.get("abstract_text", "")).strip()
        references_text = str(payload.get("references_text", "")).strip()
        final_markdown = str(payload.get("final_markdown") or "").strip()

        if not final_markdown:
            raise ValueError("finalization_agent 未返回 final_markdown，请检查整合智能体输出")

        self.runtime_state.abstract_text = abstract_text
        self.runtime_state.references_text = references_text
        self.runtime_state.final_markdown = final_markdown
        self.workspace.write_final_report(final_markdown)

    def _step_stream(self, memory_step: ActionStep) -> Generator[ToolOutput | ActionOutput, None, None]:
        try:
            self.logger.log_rule("🚀 [系统启动] 长文本生成流程开始（子智能体调度模式）", level=LogLevel.INFO)
            self._planning_phase(memory_step)
            self._writing_phase(memory_step)
            self._finalization_phase(memory_step)
            self.logger.log_rule("🎉 [系统完成] 长文本生成流程结束", level=LogLevel.INFO)
            yield ActionOutput(output=self._generate_final_output(), is_final_answer=True)
        except Exception as exc:
            self.logger.log(f"❌ [系统崩溃] LongWriterAgent 发生致命错误: {exc}", level=LogLevel.ERROR)
            raise

    def _generate_final_output(self) -> str:
        try:
            with open(self.workspace.final_output_file, "r", encoding="utf-8") as file_handle:
                content = file_handle.read()
        except (FileNotFoundError, IOError):
            content = "（最终文件生成失败，请检查日志）"

        summary_lines = [
            "# 📝 报告生成完成",
            "## 生成信息",
            f"流式文件 {self.workspace.output_file}",
            f"排版文件 {self.workspace.final_output_file}",
            f"任务摘要 {self.task[:60]}...",
            "## 预览内容",
            content[:1200] + "\n\n...完整报告见最终生成文件",
        ]
        return "\n".join(summary_lines)
