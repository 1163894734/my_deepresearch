from typing import List, Dict, Any, Optional, Generator, Tuple, Set
import json
import re
import time
import os
from pathlib import Path

from smolagents import CustomAgent
from smolagents.agents import ToolOutput, ActionOutput
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel

try:
    from .citation_validator import CitationValidator
    from .long_writer import (
        JsonWorkflowComponent,
        KeywordSearchPlanningService,
        OutlineParsingService,
        SectionWritingService,
        build_workflow_components,
    )
except ImportError:
    from citation_validator import CitationValidator
    from long_writer import (
        JsonWorkflowComponent,
        KeywordSearchPlanningService,
        OutlineParsingService,
        SectionWritingService,
        build_workflow_components,
    )


class ReportWorkspace:
    """独立的文件与状态持久化管理器，彻底剥离 Agent 的 IO 职责"""
    def __init__(self, base_dir: str, agent_logger):
        self.logger = agent_logger
        self.output_dir = base_dir
        
        self.output_file = f"{self.output_dir}/report.md"
        self.final_output_file = f"{self.output_dir}/report_final.md"
        self.log_file = f"{self.output_dir}/generation_log.txt"
        self.reference_trace_file = f"{self.output_dir}/reference_trace_log.txt"
        self.citations_validation_log = f"{self.output_dir}/citations_validation.json"
        self.tagged_search_log_file = f"{self.output_dir}/tagged_search_log.jsonl"
        
        self.max_skill_log_chars = 1200
        self._init_directories()

    def _init_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.safe_write(self.output_file, f"# 生成的长文本报告\n\n初始化时间 {timestamp}\n\n---\n\n", "w")
        self.safe_write(self.reference_trace_file, f"# 参考文献追踪日志\n\n生成时间 {timestamp}\n\n说明 记录正文中每条引用出现的章节与句子，以及引用库新增条目详情。\n\n---\n\n", "w")
        self.safe_write(self.tagged_search_log_file, f"# tagged web 检索结构化日志(JSONL)\n# 生成时间 {timestamp}\n# 每行一个 JSON 对象，便于后处理/回放\n", "w")

    def safe_write(self, file_path: str, content: str, mode: str = "a") -> None:
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.log(f"文件操作失败 {file_path} {e}", level=LogLevel.ERROR)

    def get_heading_prefix(self, section_number: str = "", section_level: Optional[int] = None) -> str:
        if not section_number:
            return "##" if section_level is None or section_level <= 1 else ("###" if section_level == 2 else "####")
        normalized = str(section_number).strip().rstrip('.')
        if not normalized:
            return "##" if section_level is None or section_level <= 1 else ("###" if section_level == 2 else "####")
        depth = normalized.count('.') + 1
        return "##" if depth <= 1 else ("###" if depth == 2 else "####")

    def filter_references(self, content: str) -> str:
        pattern = r'\n(#+\s*(参考文献|References|引用|Bibliography)|\*\*(参考文献|References)\*\*).*'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            self.logger.log("🔍 检测到参考文献部分，已从正文中移除", level=LogLevel.DEBUG)
            return content[:match.start()].strip()
        return content.strip()

    def append_section(self, title: str, content: str, section_index: int = 0, total_sections: int = 0, section_type: str = "body", section_number: str = "", section_level: Optional[int] = None):
        progress = f" [{section_index}/{total_sections}]" if section_index > 0 and total_sections > 0 else ""
        heading_prefix = self.get_heading_prefix(section_number, section_level)
        numbered_title = f"{section_number} {title}".strip() if section_number else title
        
        filtered_content = content if section_type == "references" else self.filter_references(content)
        block = f"{heading_prefix} {numbered_title}{progress}\n\n{filtered_content}\n\n"
        self.safe_write(self.output_file, block)
        self.logger.log(f"✅ 段落已保存到文件{progress} {self.output_file}", level=LogLevel.INFO)

    def save_outline(self, outline: str):
        try:
            content = f"## 📋 文章大纲\n\n{outline}\n\n---\n\n"
            self.safe_write(self.output_file, content, "a")
            self.logger.log(f"✅ 大纲已保存到文件 {self.output_file}", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log(f"⚠️ 保存大纲失败 {e}", level=LogLevel.ERROR)

    def write_final_report(self, outline: str, sections: List[Dict[str, Any]]):
        content_blocks = [f"# 生成的长文本报告 (最终排版版)\n\n## 📋 文章大纲\n\n{outline}\n\n---\n\n"]
        for section in sections:
            title = section.get("title", "未命名章节")
            section_type = section.get("type", "body")
            section_number = str(section.get("number", "")).strip()
            heading_prefix = self.get_heading_prefix(section_number, section.get("level"))
            numbered_title = f"{section_number} {title}".strip() if section_number else title
            
            raw_content = str(section.get("content", ""))
            filtered = raw_content if section_type == "references" else self.filter_references(raw_content)
            content_blocks.append(f"{heading_prefix} {numbered_title}\n\n{filtered}\n\n")
            
        self.safe_write(self.final_output_file, "".join(content_blocks), "w")

    def preview_for_log(self, value: Any) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
        except Exception:
            text = repr(value)
        text = text.strip()
        return text if len(text) <= self.max_skill_log_chars else f"{text[:self.max_skill_log_chars]}\n... (已截断，原始长度 {len(text)} 字符)"

    def log_skill_call(self, tool_name: str, arguments: Any, output: Any = None, status: str = "success", elapsed_ms: Optional[int] = None, error_message: str = "", current_step: str = "unknown", resolved_skill_key: str = "") -> None:
        try:
            input_preview = self.preview_for_log(arguments).replace('\n', ' ')
            cost_str = f" ({elapsed_ms}ms)" if elapsed_ms is not None else ""
            
            if status == "success":
                self.logger.log(f"🟢 [Skill成功] {tool_name}{cost_str} | 输入简览: {input_preview[:50]}...", level=LogLevel.INFO)
            else:
                self.logger.log(f"🔴 [Skill失败] {tool_name}{cost_str} | 错误: {error_message}", level=LogLevel.ERROR)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            full_input = json.dumps(arguments, ensure_ascii=False, indent=2) if isinstance(arguments, (dict, list)) else str(arguments)
            full_output = json.dumps(output, ensure_ascii=False, indent=2) if isinstance(output, (dict, list)) else (str(output) if output is not None else "")
            
            log_block = f"\n## 🧩 [Skill调用] {tool_name} @ {timestamp}\n"
            log_block += f"- **状态**: `{'成功' if status == 'success' else '失败'}`\n"
            log_block += f"- **耗时**: `{elapsed_ms} ms`\n"
            log_block += f"- **当前阶段**: `{current_step}`\n"
            log_block += f"- **提示词映射**: `{resolved_skill_key or '未命中'}`\n\n"
            log_block += "### 输入参数 (Input)\n" + full_input + "\n\n"
            if status == "success": log_block += "### 返回结果 (Output)\n" + full_output + "\n\n"
            else: log_block += "### 错误信息 (Error)\n" + error_message + "\n\n"
            log_block += "---\n"
            self.safe_write(self.log_file, log_block)
        except Exception as e:
            self.logger.log(f"⚠️ [系统异常] Skill日志写入失败: {e}", level=LogLevel.ERROR)

    def log_reference_event(self, stage: str, section_title: str, detail: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.safe_write(self.reference_trace_file, f"[{timestamp}] [{stage}] 章节 {section_title or '未知章节'}\n{detail}\n\n")
        
    def log_reflection(self, stage: str, score: int, report: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        content = f"\n{'='*80}\n【{stage}】 - {timestamp}\n评分 {score}\n{'-'*80}\n{report}\n{'='*80}\n\n"
        self.safe_write(self.log_file, content)
        
    def log_revision(self, stage: str, content: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_content = f"\n{'='*80}\n【{stage} - 修订结果】 - {timestamp}\n{'-'*80}\n{content}\n{'='*80}\n\n"
        self.safe_write(self.log_file, log_content)

    def append_citation_validation_text_log(self, section_type: str, section_ref: str, before_text: str, after_text: str, validation_result: Optional[Dict[str, Any]] = None, error: str = "") -> None:
        try:
            existing_data: Dict[str, Any] = {}
            if os.path.exists(self.citations_validation_log):
                with open(self.citations_validation_log, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

            section_text_logs = existing_data.get("section_text_logs", [])
            summary = {k: validation_result.get(k) for k in ["status", "total", "verified", "rejected", "verification_rate"]} if isinstance(validation_result, dict) else {}

            section_text_logs.append({
                "timestamp": time.time(), "section_type": section_type, "section_title": section_ref,
                "before_text": str(before_text or ""), "after_text": str(after_text or ""),
                "changed": str(before_text or "") != str(after_text or ""), "validation_result_summary": summary, "error": str(error or "")
            })

            existing_data["section_text_logs"] = section_text_logs
            with open(self.citations_validation_log, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.log(f"⚠️ 写入引用验证日志失败 {e}", level=LogLevel.ERROR)


class LongWriterAgent(CustomAgent):
    STATE_COARSE_RAG_CONTEXT = "coarse_rag_context"
    STATE_LATEST_TAGGED_SEARCH_PAYLOAD = "_latest_tagged_search_payload"
    STATE_TAGGED_RETRIEVAL_RECORDS = "tagged_retrieval_records"
    STATE_SECTION_TAGGED_PAYLOAD = "section_tagged_payload"
    STATE_OUTLINE = "outline"
    STATE_SECTIONS = "sections"

    def __init__(self, model, tools: Optional[List] = None, **kwargs):
        super().__init__(model=model, tools=tools or [], **kwargs)
        
        # --- 核心重构：隔离所有的 IO 与基础状态到 Workspace ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.workspace = ReportWorkspace(f"outputs/report_Frontier_Review_{timestamp}", self.logger)
        
        # 向下兼容：保留老代码组件中可能直接使用的属性
        self._output_dir = self.workspace.output_dir
        self._output_file = self.workspace.output_file
        self._final_output_file = self.workspace.final_output_file
        self._log_file = self.workspace.log_file
        self._reference_trace_file = self.workspace.reference_trace_file
        self._citations_validation_log = self.workspace.citations_validation_log
        self._tagged_search_log_file = self.workspace.tagged_search_log_file
        
        # 全局缓冲与状态
        self._current_outline = ""
        self._generated_sections: List[Dict[str, str]] = []
        self._abstract_section: Optional[Dict[str, str]] = None
        self._previous_section_content = ""
        self._prev_body_or_intro_content = ""
        self._global_summary = ""
        self._full_text_body = ""
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._citation_counter = 0

        # 配置参数
        self.state["search_engine"] = "openalex"
        self.state["search_sort"] = "citation"
        
        self._citation_validator = CitationValidator(model=model,remove_entire_invalid_sentence=False)
        self._unverified_citations: List[Tuple[str, str, str, str]] = []
        self._queued_citation_keys: Set[str] = set()
        self.enable_citation_validation = True
        self.citation_validation_timeout = 5
        self.allow_add_citation_from_generated_text = False
        self.strict_citation_flow = True
        self.strict_citation_revision_max_attempts = 1
        self.debug_mode = True
        self.debug_max_body_sections = 1
        self.body_generated_count = 0
        
        self.outline_max_iter = 1
        self.section_max_iter = 1
        self.outline_score_threshold = 90
        self.section_score_threshold = 88
        self.max_summary_length = 300
        self.search_tool_name = "web_search"
        self.max_outline_context_chars = 1200
        self.max_prev_section_chars = 900
        self.max_fine_rag_chars = 1600
        self.max_bibliography_chars = 2200
        
        self.workspace.max_skill_log_chars = 1200 # 同步给 workspace
        self._current_step = "init"
        
        self.enable_coarse_rag_web_search = True
        self.coarse_rag_mode = 'web_agent'
        self.enable_fine_rag_web_search = True
        self.fine_rag_mode = 'web_agent'
        self.text_webbrowser_agent_name = "custom_search_agent"
        self.tagged_search_top_k = 8
        self._tagged_retrieval_records: List[Dict[str, Any]] = []
        
        # 服务装配
        self._keyword_search_service = KeywordSearchPlanningService()
        self._outline_parsing_service = OutlineParsingService()
        self._section_writing_service = SectionWritingService()
        self._workflow_components = self._build_workflow_components()
        self.state["workflow_component_contracts"] = self.get_workflow_component_contracts()
        
        self._skill_prompts: Dict[str, str] = {}
        self._skill_name_aliases: Dict[str, str] = {}
        self._cache_skill_prompts()

    # ==========================
    # IO 方法委托 (Delegation)
    # ==========================
    def _safe_write_file(self, *args, **kwargs): self.workspace.safe_write(*args, **kwargs)
    def _append_section_to_file(self, *args, **kwargs): self.workspace.append_section(*args, **kwargs)
    def _get_heading_prefix(self, *args, **kwargs): return self.workspace.get_heading_prefix(*args, **kwargs)
    def _save_outline_to_file(self, *args, **kwargs): self.workspace.save_outline(*args, **kwargs)
    def _write_final_output_file(self, *args, **kwargs): self.workspace.write_final_report(*args, **kwargs)
    def _log_reference_event(self, *args, **kwargs): self.workspace.log_reference_event(*args, **kwargs)
    def _append_citation_validation_text_log(self, *args, **kwargs): self.workspace.append_citation_validation_text_log(*args, **kwargs)
    def _filter_references_from_content(self, *args, **kwargs): return self.workspace.filter_references(*args, **kwargs)
    def _log_reflection_to_file(self, *args, **kwargs): self.workspace.log_reflection(*args, **kwargs)
    def _log_revision_to_file(self, *args, **kwargs): self.workspace.log_revision(*args, **kwargs)

    def set_text_webbrowser_agent_name(self, agent_name: str):
        self.text_webbrowser_agent_name = agent_name
        self.logger.log(f"✅ text_webbrowser_agent 已配置 {agent_name}", level=LogLevel.INFO)

    def _cache_skill_prompts(self) -> None:
        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.exists():
            self.logger.log(f"⚠️ Skills目录不存在 {skills_dir}", level=LogLevel.INFO)
            return
        
        count = 0
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    lines = content.splitlines()
                    metadata: Dict[str, str] = {}
                    if lines and lines[0].strip() == "---":
                        end_idx = None
                        for i in range(1, len(lines)):
                            if lines[i].strip() == "---":
                                end_idx = i
                                break
                        if end_idx is not None:
                            for raw_line in lines[1:end_idx]:
                                if ":" not in raw_line:
                                    continue
                                k, v = raw_line.split(":", 1)
                                metadata[str(k).strip()] = str(v).strip().strip('"\'')
                            prompt_body = "\n".join(lines[end_idx + 1:]).lstrip()
                            skill_name = skill_dir.name
                            self._skill_prompts[skill_name] = prompt_body
                            frontmatter_name = str(metadata.get("name", "")).strip()
                            if frontmatter_name:
                                self._skill_name_aliases[frontmatter_name] = skill_name
                                self._skill_name_aliases[self._normalize_skill_lookup_key(frontmatter_name)] = skill_name
                            self._skill_name_aliases[self._normalize_skill_lookup_key(skill_name)] = skill_name
                            count += 1
                except Exception:
                    pass
        
        self.logger.log(f"✅ 已缓存 {count} 个skill的提示词", level=LogLevel.DEBUG)

    def _normalize_skill_lookup_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", str(value or "").strip().lower())

    def _resolve_skill_prompt(self, tool_name: str) -> Tuple[str, str]:
        if tool_name in self._skill_prompts:
            return self._skill_prompts[tool_name], tool_name
        alias_key = self._skill_name_aliases.get(tool_name)
        if alias_key and alias_key in self._skill_prompts:
            return self._skill_prompts[alias_key], alias_key
        normalized_name = self._normalize_skill_lookup_key(tool_name)
        alias_key = self._skill_name_aliases.get(normalized_name)
        if alias_key and alias_key in self._skill_prompts:
            return self._skill_prompts[alias_key], alias_key
        if normalized_name.endswith("_write"):
            writer_key = normalized_name[:-6] + "_writer"
            alias_key = self._skill_name_aliases.get(writer_key, writer_key)
            if alias_key in self._skill_prompts:
                return self._skill_prompts[alias_key], alias_key
        return "", ""

    def _build_workflow_components(self) -> Dict[str, JsonWorkflowComponent]:
        return build_workflow_components()

    def _invoke_workflow_component(self, component_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log(f"🔄 [组件调用] 开始执行: {component_name}", level=LogLevel.INFO)
        start_time = time.perf_counter()
        
        component = self._workflow_components.get(component_name)
        try:
            # ✅ BUG FIXED: 传入 self (Agent实例本身) 而非 self.state
            result = component.run(self, payload) 
            cost = int((time.perf_counter() - start_time) * 1000)
            
            if component_name == "academic_search":
                engine = result.get("engine_used", "未知")
                count = len(result.get("available_citations", {}))
                self.logger.log(f"✅ [检索完成] 使用 {engine} 引擎，获取到 {count} 篇文献 (耗时: {cost}ms)", level=LogLevel.INFO)
            elif component_name == "keyword_search_expansion":
                kws = result.get("candidate_keywords", [])
                self.logger.log(f"✅ [关键词扩展] 挖掘出新候选词: {kws} (耗时: {cost}ms)", level=LogLevel.INFO)
            else:
                self.logger.log(f"✅ [组件完成] {component_name} (耗时: {cost}ms)", level=LogLevel.INFO)
                
            return result
        except Exception as e:
            cost = int((time.perf_counter() - start_time) * 1000)
            self.logger.log(f"❌ [组件失败] {component_name} 异常中断 (耗时: {cost}ms): {e}", level=LogLevel.ERROR)
            raise

    def get_workflow_component_contracts(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: component.contract()
            for name, component in self._workflow_components.items()
        }

    def _clean_task_description(self) -> str:
        task = self.task
        task_marker = "Task:\n"
        if task_marker in task:
            task_content = task[task.find(task_marker) + len(task_marker):]
            if "\n---\n" in task_content:
                task_content = task_content.split("\n---\n")[0]
            return task_content.strip()
        return task

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self.logger.log(f"🛠️ [Skill准备] 准备调用: {tool_name}", level=LogLevel.DEBUG)
        start = time.perf_counter()
        try:
            result = super().execute_tool_call(tool_name, arguments)
            cost = int((time.perf_counter() - start) * 1000)
            skill_prompt, resolved_skill_key = self._resolve_skill_prompt(tool_name)
            self.workspace.log_skill_call(tool_name, arguments, result, "success", cost, current_step=self._current_step, resolved_skill_key=resolved_skill_key)
            return result
        except Exception as e:
            cost = int((time.perf_counter() - start) * 1000)
            skill_prompt, resolved_skill_key = self._resolve_skill_prompt(tool_name)
            self.workspace.log_skill_call(tool_name, arguments, None, "failed", cost, str(e), current_step=self._current_step, resolved_skill_key=resolved_skill_key)
            raise
    
    def _step_stream(self, memory_step: ActionStep) -> Generator[ToolOutput | ActionOutput, None, None]:
        try:
            self.logger.log_rule("🚀 [系统启动] 长文本生成流程开始", level=LogLevel.INFO)
            
            self._current_step = "planner"
            self._planning_phase(memory_step)
            
            self._current_step = "writer"
            self._writing_phase(memory_step)
            
            self._current_step = "finalizer"
            self._finalization_phase(memory_step)
            
            self.logger.log_rule("🎉 [系统完成] 长文本生成流程结束", level=LogLevel.INFO)
            yield ActionOutput(output=self._generate_final_output(), is_final_answer=True)
            
        except Exception as e:
            self.logger.log(f"❌ [系统崩溃] LongWriterAgent 发生致命错误: {e}", level=LogLevel.ERROR)
            raise
    
    def _classify_section_type(self, title: str, index: int, total: int) -> str:
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["摘要", "abstract", "executive summary"]): return "abstract"
        if any(kw in title_lower for kw in ["参考文献", "references", "bibliography"]): return "references"
        if any(kw in title_lower for kw in ["结论", "总结", "conclusion", "summary", "展望", "future"]): return "conclusion"
        if index == 0 or any(kw in title_lower for kw in ["引言", "介绍", "introduction", "背景", "background"]): return "introduction"
        return "body"

    def _outline_reflection_loop(self, initial_outline: str, payload: dict) -> str:
        current = initial_outline
        for i in range(self.outline_max_iter):
            self.logger.log(f"  🔄 [大纲反思] 第 {i+1}/{self.outline_max_iter} 轮", level=LogLevel.DEBUG)
            try:
                reflection_input = f"大纲:\n{current}\n参考资料:\n{payload}"
                report_raw = self.execute_tool_call("outline_reflection", {"input": reflection_input})
                report = str(report_raw)
                
                from utils.common_utils import safe_json_parse
                data = safe_json_parse(report)
                if not isinstance(data, dict): data = {}
                
                score = data.get("score", 0)
                is_pass = data.get("is_pass", False)
                self._log_reflection_to_file(f"大纲反思第{i+1}轮", score, report)

                if is_pass or score > self.outline_score_threshold:
                    self.logger.log(f"  ✅ [大纲通过] 得分 {score}", level=LogLevel.INFO)
                    return current

                self.logger.log(f"  ⚠️ [需要修订] 得分 {score} 未达标 ({self.outline_score_threshold})", level=LogLevel.INFO)
                revision_prompt = f"大纲:\n{current}\n评审报告:\n{report}\n参考资料:\n{payload}"
                current = str(self.execute_tool_call("outline_revision", {"input": revision_prompt}))
                self._log_revision_to_file(f"大纲修订第{i+1}轮", current)

            except Exception as e:
                self.logger.log(f"  ❌ 反思出错中断: {e}", level=LogLevel.ERROR)
                return current
        return current

    def _section_reflection_loop(self, text: str, section: dict, available_citations: dict) -> str:
        current = text
        for i in range(self.section_max_iter):
            self.logger.log(f"  🔍 [段落反思] 章节《{section.get('title', '')}》", level=LogLevel.DEBUG)
            try:
                import json
                evidence_reflection_input = {"section": section, "content": current, "available_citations": available_citations}
                report_raw = self.execute_tool_call("evidence_reflection", {"input": json.dumps(evidence_reflection_input, ensure_ascii=False)})
                report = str(report_raw)
                self._log_reflection_to_file(f"证据反思 - {section.get('goal', '')[:30]}", 0, report)

                from utils.common_utils import safe_json_parse
                section_revision_input = {
                    "section": section,
                    "original_content": current,
                    "reflection_report": safe_json_parse(report),
                }
                current = str(self.execute_tool_call("section_revision", {"input": json.dumps(section_revision_input, ensure_ascii=False)}))
                self._log_revision_to_file(f"段落修订 - {section.get('goal', '')[:30]}", current)
                self.logger.log("  ✅ 段落修订完成", level=LogLevel.INFO)

            except Exception as e:
                self.logger.log(f"  ❌ 反思出错中断: {e}", level=LogLevel.ERROR)
                return current
        return current

    def _planning_phase(self, memory_step: ActionStep) -> None:
        self.logger.log_rule("🗺️ [阶段 1/3] 规划与大纲生成 (Planning Phase)", level=LogLevel.INFO)
        cleaned_task = self._clean_task_description()
        
        self.logger.log("🔍 [阶段 1] 开始执行文献检索...", level=LogLevel.INFO)
        search_stage_result = self._invoke_workflow_component("keyword_search_expansion", {"task": cleaned_task})
        combined_citations = search_stage_result.get("available_citations", {})
        
        self.logger.log("📝 [阶段 1] 基于检索结果生成大纲...", level=LogLevel.INFO)
        outline_stage_result = self._invoke_workflow_component("outline_generation", {
            "task": cleaned_task,
            "available_citations": combined_citations
        })
        outline_v1 = outline_stage_result.get("outline_v1", "")
        self._current_outline = self._outline_reflection_loop(outline_v1, outline_stage_result.get("outline_input", {}))
        
        self._citations.update(combined_citations)
        self._citations.update(outline_stage_result.get("available_citations", {}))
        
        self.logger.log(f"✅ [阶段 1] 大纲规划完成，已将 {len(combined_citations)} 篇前沿文献载入全局记忆。", level=LogLevel.INFO)
        self.workspace.save_outline(self._current_outline)
    
    def _writing_phase(self, memory_step: ActionStep) -> None:
        def parse_logger(msg: str, level_str: str):
            from smolagents.monitoring import LogLevel
            level_map = {"DEBUG": LogLevel.DEBUG, "INFO": LogLevel.INFO, "ERROR": LogLevel.ERROR}
            self.logger.log(msg, level=level_map.get(level_str, LogLevel.INFO))
            
        sections = self._outline_parsing_service.parse_sections(
            outline=self._current_outline, 
            log_callable=parse_logger
        )
        self.logger.log_rule("✍️ [阶段 2/3] 分发与撰写 (Writing Phase)", level=LogLevel.INFO)
        self._prev_body_or_intro_content = ""
        total_sections = len(sections)
        self.logger.log(f"📊 [阶段 2] 大纲解析完毕，共拆分为 {total_sections} 个章节", level=LogLevel.INFO)

        for idx, section in enumerate(sections):
            current_section_num = idx + 1
            section_type = self._classify_section_type(section['title'], idx, len(sections))
            self.logger.log_rule(f"📌 [段落 {current_section_num}/{total_sections}] {section['title']} (类型: {section_type})", level=LogLevel.INFO)

            if section_type == "abstract":
                self._abstract_section = section
                self.logger.log("⏭️ [段落跳过] 摘要章节将延后至生成结束时撰写", level=LogLevel.INFO)
                continue

            if not section.get("is_leaf", True):
                self.logger.log("⏭️ [段落跳过] 检测为父节点目录，仅占位不生成正文", level=LogLevel.INFO)
                self._generated_sections.append({
                    "title": section['title'], "content": "", "type": section_type,
                    "number": section.get("number", ""), "level": section.get("level")
                })
                self.workspace.append_section(section['title'], "", current_section_num, total_sections, section_type, section.get('number', ''), section.get('level'))
                continue

            fine_rag_context = ""
            available_citations = {}
            if section_type == "body":
                if self.debug_mode and self.body_generated_count >= self.debug_max_body_sections:
                    self.logger.log("🛑 [调试模式] 已达到最大正文段落数限制", level=LogLevel.INFO)
                    break
                self.body_generated_count += 1
                self.logger.log(f"🔎 [资料检索] 开始针对章节主体进行细粒度检索...", level=LogLevel.INFO)
                
                fine_rag_context = self._run_fine_rag_for_section(section)
                if fine_rag_context:
                    try:
                        parsed = json.loads(fine_rag_context)
                        available_citations = parsed if isinstance(parsed, dict) else {}
                        self.logger.log(f"✅ [资料检索] 获取到 {len(available_citations)} 条相关引用", level=LogLevel.INFO)
                    except (json.JSONDecodeError, TypeError):
                        self.logger.log("⚠️ [资料检索异常] 细粒度资料检索返回结果无法解析为JSON，已回退为空", level=LogLevel.ERROR)
                        available_citations = {}

            cleaned_task = self._clean_task_description()
            final = ""
            
            if section_type == "references":
                final = str(self._invoke_workflow_component("references_writing", {
                    "section": section, "available_citations": self._citations, "task": cleaned_task, "outline": self._current_outline
                }).get("content", ""))
                
            elif section_type == "introduction":
                payload = {
                    "section": section, "available_citations": self._citations.copy(), 
                    "task": cleaned_task, "prev_section_content": self._prev_body_or_intro_content, "outline": self._current_outline
                }
                self.logger.log("✍️ [引言撰写] 开始生成引言...", level=LogLevel.INFO)
                final = SectionWritingService.write_intro(self, payload)
                self.logger.log("✅ [引言撰写] 引言生成完毕", level=LogLevel.INFO)
                
                self.logger.log(f"🛡️ [引用校验] 开始校验 {section['title']} 的引用规范...", level=LogLevel.INFO)
                available_citations, final = self._citation_validator.run_five_step_validation(self._citations.copy(), final, self._citations_validation_log, section.get("title", "引言"))
                self._prev_body_or_intro_content = final
                
            elif section_type == "conclusion":
                payload = {
                    "section": section, "available_citations": self._citations.copy(), 
                    "full_text": self._full_text_body, "task": cleaned_task, "outline": self._current_outline
                }
                self.logger.log("✍️ [结论撰写] 开始生成结论...", level=LogLevel.INFO)
                final = SectionWritingService.write_conclusion(self, payload)
                self.logger.log("✅ [结论撰写] 结论生成完毕", level=LogLevel.INFO)
                
                self.logger.log(f"🛡️ [引用校验] 开始校验 {section['title']} 的引用规范...", level=LogLevel.INFO)
                available_citations, final = self._citation_validator.run_five_step_validation(self._citations.copy(), final, self._citations_validation_log, section.get("title", "结论"))
                self._prev_body_or_intro_content = final
                
            else:
                if self.debug_mode and self.body_generated_count > self.debug_max_body_sections:
                    self.logger.log("🛑 [调试模式] 已达到最大正文段落数限制，跳过剩余正文生成", level=LogLevel.INFO)
                    break
                payload = {
                    "section": section, "available_citations": available_citations, 
                    "fine_rag_context": fine_rag_context, "task": cleaned_task, 
                    "prev_section_content": self._prev_body_or_intro_content, "outline": self._current_outline    
                }
                
                self.logger.log("🏗️ [正文撰写] 步骤 1/4 - 骨架规划...", level=LogLevel.INFO)
                skeleton = SectionWritingService.plan_body_skeleton(self, payload)
                self.logger.log("✅ [正文撰写] 骨架规划完毕", level=LogLevel.INFO)
                
                self.logger.log("✍️ [正文撰写] 步骤 2/4 - 文本成稿...", level=LogLevel.INFO)
                payload["skeleton"] = skeleton
                final = SectionWritingService.compose_body_content(self, payload)
                self.logger.log("✅ [正文撰写] 文本组装完毕", level=LogLevel.INFO)
                
                self.logger.log(f"🛡️ [正文撰写] 步骤 3/4 - 校验引用规范...", level=LogLevel.INFO)
                available_citations, final = self._citation_validator.run_five_step_validation(available_citations, final, self._citations_validation_log, section.get("title", "正文"))
                
                self.logger.log(f"🔍 [正文撰写] 步骤 4/4 - 执行反思修订...", level=LogLevel.INFO)
                final = self._section_reflection_loop(final, section, available_citations)
                
                self._prev_body_or_intro_content = final
                self._citations.update(available_citations)
            
            if section_type != "references":
                self._update_memory(section['title'], final)
            
            self._generated_sections.append({
                "title": section['title'], "content": final, "type": section_type,
                "number": section.get("number", ""), "level": section.get("level")
            })
            self.workspace.append_section(section['title'], final, current_section_num, total_sections, section_type, section.get('number', ''), section.get('level'))
            self.logger.log(f"🎉 [段落完成] {section['title']} 生成完毕 (字数: {len(str(final))})", level=LogLevel.INFO)

    def _run_fine_rag_for_section(self, section: Dict[str, str]) -> str:
        section_title = section.get('title', '未命名章节')
        self.state[self.STATE_LATEST_TAGGED_SEARCH_PAYLOAD] = None
        if self.fine_rag_mode == 'disabled': return ""
        
        search_query = f"{section_title} {section.get('goal', '')}".strip()
        if self.fine_rag_mode == 'web_agent':
            tagged_result = self._invoke_workflow_component("academic_search", {"task": section_title})
            return json.dumps(tagged_result.get("available_citations"), ensure_ascii=False)
        elif self.fine_rag_mode == 'web_search':
            self.logger.log(f"🔎 [资料检索] 调用 web_search 检索: {search_query}", level=LogLevel.DEBUG)
            try:
                result = self._section_writing_service.run_fine_rag_web_search(self, section_title, search_query)
                self.logger.log(f"✅ [资料检索] web_search成功 ({len(result)} 字)", level=LogLevel.DEBUG)
                return result
            except Exception as e:
                self.logger.log(f"⚠️ [资料检索] web_search调用失败: {e}", level=LogLevel.ERROR)
                raise
        raise ValueError(f"未知的细粒度资料检索模式 {self.fine_rag_mode}")

    def _log_reference_usage_in_section(self, section_title: str, text: str) -> None:
        if not text: return
        sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])\s+|\n+', str(text)) if s.strip()]
        total_usage = 0
        for idx, sentence in enumerate(sentences, 1):
            for author_raw, year_raw in re.findall(r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)", sentence):
                self.workspace.log_reference_event("citation_used", section_title, f"引用键 {author_raw.strip()} ({year_raw.strip()})\n句子序号 第{idx}句\n句子内容 {sentence}")
                total_usage += 1
        if total_usage == 0:
            self.workspace.log_reference_event("citation_used", section_title, "本章节未检测到 APA 文内引用。")

    def _log_reference_addition(self, source: str, section_title: str, citation_key: str, citation_info: Dict[str, Any]) -> None:
        try:
            info_text = json.dumps(citation_info, ensure_ascii=False, indent=2)
        except Exception:
            info_text = str(citation_info)
        self.workspace.log_reference_event("citation_added", section_title, f"来源 {source}\n入库键 {citation_key}\n入库详情\n{info_text}")

    def enable_citation_validation_mode(self, enable: bool = True):
        self.enable_citation_validation = enable
        if enable: self.logger.log("✅ 启用学术引用验证系统", level=LogLevel.INFO)

    def _finalization_phase(self, memory_step: ActionStep) -> None:
        self.logger.log_rule("🎁 [阶段 3/3] 整合与后处理 (Finalization Phase)", level=LogLevel.INFO)
        
        if self.enable_citation_validation:
            self.logger.log("🔄 [阶段 3] 刷新全局参考文献列表...", level=LogLevel.INFO)
            self._refresh_references_sections_content()

        if not self._abstract_section:
            self.logger.log("⚠️ [阶段 3] 大纲中未检测到摘要章节，触发自动补充逻辑", level=LogLevel.INFO)
            self._abstract_section = {
                "title": "摘要", "type": "abstract", "number": "", "level": 1, "is_leaf": True
            }

        if self._abstract_section:
            self.logger.log("📝 [阶段 3] 开始基于全局全文生成最终摘要...", level=LogLevel.INFO)
            cleaned_task = self._clean_task_description()
            
            payload = {
                "section": self._abstract_section, "full_text": self._full_text_body, "task": cleaned_task
            }
            final_abstract = SectionWritingService.write_abstract(self, payload)
            self.logger.log("✅ [阶段 3] 摘要生成完毕", level=LogLevel.INFO)
            
            self._abstract_section["content"] = final_abstract
            
            self._generated_sections = [s for s in self._generated_sections if s.get("type") != "abstract"]
            intro_index = next((i for i, s in enumerate(self._generated_sections) if s.get("type") == "introduction"), 0)
            self._generated_sections.insert(intro_index, self._abstract_section)
            self.logger.log("✅ [阶段 3] 摘要已成功插入", level=LogLevel.INFO)

        self.logger.log("💾 [文件输出] 开始写入最终排版文件...", level=LogLevel.INFO)
        self.workspace.write_final_report(self._current_outline, self._generated_sections)
        
        self.state[self.STATE_OUTLINE] = self._current_outline
        self.state[self.STATE_SECTIONS] = self._generated_sections
        self.logger.log("✅ [阶段 3] 整合排版结束", level=LogLevel.INFO)

    def _get_section_ref(self, section: Dict[str, Any]) -> str:
        number = str(section.get("number", "")).strip()
        title = str(section.get("title", "未知章节")).strip() or "未知章节"
        return f"{number} {title}".strip() if number else title
    
    def _update_memory(self, title: str, content: str):
        self._previous_section_content = content
        block_content = str(content or "").strip()
        if block_content:
            block = f"## {str(title or '').strip() or '未命名章节'}\n\n{block_content}"
            self._full_text_body = f"{self._full_text_body}\n\n{block}" if self._full_text_body else block
        
        combined = (self._global_summary + "\n" + content) if self._global_summary else content
        sentences = [s.strip() for s in combined.replace("。", "。\n").split("\n") if s.strip()]
        
        summary = "。".join([sentences[0]] + [sentences[i] for i in range(1, len(sentences), max(1, len(sentences)//3))]) if len(sentences) > 3 else combined
        self._global_summary = summary[:self.max_summary_length] + "..." if len(summary) > self.max_summary_length else summary

    def _format_references_section(self) -> str:
        if not self._citations: return "暂无引用文献。"
        sorted_refs = sorted(self._citations.items(), key=lambda x: (str(x[1].get('authors', '')).lower(), str(x[1].get('year', ''))))
        return "\n".join([f"[{idx}] {info.get('authors', '未知作者')}. ({info.get('year', 'n.d.')}). {info.get('title', 'Title unavailable')}." for idx, (_, info) in enumerate(sorted_refs, 1)])

    def _refresh_references_sections_content(self) -> None:
        if not self._generated_sections: return
        new_content = self._format_references_section()
        for section in self._generated_sections:
            if section.get('type') == 'references': section['content'] = new_content

    def _generate_final_output(self) -> str:
        try:
            with open(self.workspace.final_output_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            content = "（最终文件生成失败，请检查日志）"
        
        summary_lines = [
            "# 📝 报告生成完成",
            "## 生成信息",
            f"流式文件 {self.workspace.output_file}",
            f"排版文件 {self.workspace.final_output_file}",
            f"任务摘要 {self.task[:60]}...",
            "## 预览内容",
            content[:1200] + "\n\n...完整报告见最终生成文件"
        ]
        
        return "\n".join(summary_lines)