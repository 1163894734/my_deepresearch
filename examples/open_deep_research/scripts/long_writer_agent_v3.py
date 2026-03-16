"""
LongWriterAgent V3 - 多阶段长文本生成代理（基于关键词的精准检索）

核心设计：
✅ 继承 CustomAgent 的完整能力（run(), memory, logger, execute_tool_call）
✅ 重写 _step_stream() 实现多阶段流程（规划 → 写作 → 整合）
✅ 多轮反思循环（大纲和每段都支持）
✅ 严格的上下文控制（只保留前一段+摘要）
✅ 内置文献管理（自动编号，避免重复）
✅ 证据达标机制（段落必须满足证据要求才停止）
✅ 学术引用验证系统（5步验证流程：Search→Verify→Retrieve→Validate→Add）
✅ 【新】基于关键词的5步精准检索流程（概念拆解 → First-Shot检索 → 术语提取 → 概念升级 → Second-Shot检索）

关键：不重新实现 Agent 循环，只自定义 _step_stream() 的处理逻辑

【V3 更新说明】
规划阶段不再使用"问题分解"方式，改用"关键词/概念组"方式：
1. 概念拆解：将研究主题拆解为2-3个核心概念组（每组3-5个关键词）
2. First-Shot检索：使用初始概念组进行第一次检索，获取Top 20文献
3. 术语提取：从文献摘要中提取高频专业术语
4. 概念升级：将高频术语归类到对应概念组，升级检索式
5. Second-Shot检索：使用升级后的概念组进行精准检索
6. 大纲生成：基于精准检索结果生成大纲

"""

from typing import List, Dict, Any, Optional, Generator, Tuple, Set
import json
import re
import time

from smolagents import CustomAgent
from smolagents.agents import ToolOutput, ActionOutput
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel

# 导入引用验证系统
try:
    from .citation_validator import CitationValidator
    from .long_writer import (
        CitationFlowService,
        JsonWorkflowComponent,
        KeywordSearchPlanningService,
        OutlineParsingService,
        SectionWritingService,
        TaggedSearchService,
        build_workflow_components,
    )
except ImportError:
    # 兼容直接脚本运行场景
    from citation_validator import CitationValidator
    from long_writer import (
        CitationFlowService,
        JsonWorkflowComponent,
        KeywordSearchPlanningService,
        OutlineParsingService,
        SectionWritingService,
        TaggedSearchService,
        build_workflow_components,
    )


class LongWriterAgent(CustomAgent):
    """
    多阶段长文本写作代理
    
    🎯 职责：
    1. 控制完整的写作流程（规划 → 写作）
    2. 执行多轮反思循环（保证质量）
    3. 管理内存和文献（内部管理）
    4. 调用各个 Skill 完成具体任务
    
    🏗️ 扩展策略：
    - 继承 CustomAgent 的 run(), memory, logger, execute_tool_call
    - 重写 _step_stream() 实现多阶段流程
    - 内部维护 outline, sections, citations
    """

    # ============== State Keys（统一收口，避免魔法字符串散落） ==============
    STATE_COARSE_RAG_CONTEXT = "coarse_rag_context"
    STATE_LATEST_TAGGED_SEARCH_PAYLOAD = "_latest_tagged_search_payload"
    STATE_TAGGED_RETRIEVAL_RECORDS = "tagged_retrieval_records"
    STATE_SECTION_TAGGED_PAYLOAD = "section_tagged_payload"
    STATE_CITATION_VALIDATION = "citation_validation"
    STATE_OUTLINE = "outline"
    STATE_SECTIONS = "sections"

    def __init__(self, model, tools: Optional[List] = None, **kwargs):
        super().__init__(model=model, tools=tools or [], **kwargs)
        
        # 流程状态
        self._current_outline = ""
        self._generated_sections: List[Dict[str, str]] = []
        self._abstract_section: Optional[Dict[str, str]] = None
        
        # 内存管理
        self._previous_section_content = ""
        self._prev_body_or_intro_content = ""  # 实时维护前一段正文或引言内容
        self._global_summary = ""
        
        # 文献管理
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._citation_counter = 0
        
        # 学术引用验证系统
        self._citation_validator = CitationValidator(model=model)
        self._unverified_citations: List[Tuple[str, str, str, str]] = []  # (author, year, title, claim)
        self._queued_citation_keys: Set[str] = set()
        self.enable_citation_validation = False  # 默认禁用（API调用较多）
        self.citation_validation_timeout = 5  # API超时时间
        # 是否允许“仅从正文文本”新增引用（默认关闭，避免虚构引用污染参考文献库）
        self.allow_add_citation_from_generated_text = False
        # 强引用流：仅允许结构化检索结果进入白名单，并在写作后进行强约束修正
        self.strict_citation_flow = True
        self.strict_citation_revision_max_attempts = 1
        
        # 输出文件管理（实时保存）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._output_dir = f"outputs/report_{timestamp}"
        self._output_file = f"{self._output_dir}/report.md"
        self._log_file = f"{self._output_dir}/generation_log.txt"
        self._reference_trace_file = f"{self._output_dir}/reference_trace_log.txt"
        self._citations_validation_log = f"{self._output_dir}/citations_validation.json"
        self._tagged_search_log_file = f"{self._output_dir}/tagged_search_log.jsonl"
        self._ensure_output_dir()
        
        # 配置（可调）
        self.outline_max_iter = 3
        self.section_max_iter = 1
        self.outline_score_threshold = 90
        self.section_score_threshold = 88
        self.max_summary_length = 300
        self.search_tool_name = "web_search"  # 使用 DuckDuckGoSearchTool 的名称
        self.max_outline_context_chars = 1200
        self.max_prev_section_chars = 900
        self.max_fine_rag_chars = 1600
        self.max_bibliography_chars = 2200
        self.max_skill_log_chars = 1200
        self._current_step = "init"
        
        # 粗细力度RAG控制开关
        self.enable_coarse_rag_web_search = True  # 是否启用粗细力度RAG的web_search
        # 可用模式: 'web_agent' (调用改进版检索代理), 'web_search' (web搜索), 'knowledge' (仅模型知识), 'disabled' (不使用粗细RAG)
        self.coarse_rag_mode = 'web_agent'
        
        # 细粒度RAG配置（调用text_webbrowser_agent进行深度检索）
        self.enable_fine_rag_web_search = True  # 是否启用细粒度RAG的web检索
        # 可用模式: 'web_agent' (调用改进版检索代理), 'web_search' (DuckDuckGo搜索), 'disabled' (禁用)
        self.fine_rag_mode = 'web_agent'
        self.text_webbrowser_agent_name = "search_agent"  # text_webbrowser_agent 的名称
        self.tagged_search_top_k = 8
        self._tagged_retrieval_records: List[Dict[str, Any]] = []
        self._citation_flow_service = CitationFlowService()
        self._keyword_search_service = KeywordSearchPlanningService()
        self._outline_parsing_service = OutlineParsingService()
        self._section_writing_service = SectionWritingService()
        self._tagged_search_service = TaggedSearchService()
        self._workflow_components = self._build_workflow_components()
        self.state["workflow_component_contracts"] = self.get_workflow_component_contracts()
    
    def set_text_webbrowser_agent_name(self, agent_name: str):
        """设置text_webbrowser_agent的名称（用于调用managed agent）
        
        Args:
            agent_name: managed agent 的名称，默认为 "search_agent"
        """
        self.text_webbrowser_agent_name = agent_name
        self.logger.log(f"✅ text_webbrowser_agent 已配置: {agent_name}", level=LogLevel.INFO)

    def _build_workflow_components(self) -> Dict[str, JsonWorkflowComponent]:
        """构建工作流组件注册表。"""
        return build_workflow_components()

    def _invoke_workflow_component(self, component_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """调用指定工作流组件，输入/输出均为 dict。"""
        component = self._workflow_components.get(component_name)
        if component is None:
            raise ValueError(f"Unknown workflow component: {component_name}")
        return component.run(self, payload)

    def get_workflow_component_contracts(self) -> Dict[str, Dict[str, Any]]:
        """返回所有组件的输入/输出格式说明。"""
        return {
            name: component.contract()
            for name, component in self._workflow_components.items()
        }
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        import os
        os.makedirs(self._output_dir, exist_ok=True)
        # 初始化输出文件
        with open(self._output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 生成的长文本报告\n\n")
            f.write(f"生成时间: {self.task}\n\n")
            f.write("---\n\n")

        # 初始化参考文献追踪日志（与 generation_log 区分）
        with open(self._reference_trace_file, 'w', encoding='utf-8') as f:
            f.write("# 参考文献追踪日志\n\n")
            f.write(f"生成时间: {self.task}\n\n")
            f.write("说明：记录正文中每条引用出现的章节与句子，以及引用库新增条目详情。\n\n")
            f.write("---\n\n")

        # 初始化 tagged 检索结构化日志
        with open(self._tagged_search_log_file, 'w', encoding='utf-8') as f:
            f.write("# tagged web 检索结构化日志(JSONL)\n")
            f.write(f"# 生成时间: {self.task}\n")
            f.write("# 每行一个 JSON 对象，便于后处理/回放\n")
    
    def _append_section_to_file(
        self,
        section_title: str,
        content: str,
        section_index: int = 0,
        total_sections: int = 0,
        section_type: str = "body",
        section_number: str = "",
        section_level: Optional[int] = None,
    ):
        """将段落内容追加到输出文件
        
        Args:
            section_title: 段落标题
            content: 段落内容
            section_index: 当前段落序号 (1-based)
            total_sections: 总段落数
            section_type: 段落类型 ("body", "references", "introduction", "conclusion")
        """
        try:
            with open(self._output_file, 'a', encoding='utf-8') as f:
                # 添加进度标识
                progress = f" [{section_index}/{total_sections}]" if section_index > 0 and total_sections > 0 else ""
                heading_prefix = self._get_heading_prefix(section_number, section_level)
                numbered_title = f"{section_number} {section_title}".strip() if section_number else section_title
                f.write(f"{heading_prefix} {numbered_title}{progress}\n\n")
                
                # 非参考文献段落：过滤掉参考文献内容
                if section_type != "references":
                    # 去除参考文献部分（如果有）
                    filtered_content = self._filter_references_from_content(content)
                    f.write(filtered_content)
                else:
                    # 参考文献段落正常输出
                    f.write(content)
                    
                f.write("\n\n")
            
            self.logger.log(f"✅ 段落已保存到文件{progress}: {self._output_file}", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log(f"⚠️ 保存段落失败: {e}", level=LogLevel.ERROR)

    def _get_heading_prefix(self, section_number: str = "", section_level: Optional[int] = None) -> str:
        """根据章节编号返回 Markdown 标题层级。

        约定：
        - 一级（如 2） -> ##
        - 二级（如 2.1） -> ###
        - 三级及更深（如 2.1.1） -> ####
        - 无编号 -> ##（兜底）
        """
        if not section_number:
            if section_level is not None:
                if section_level <= 1:
                    return "##"
                if section_level == 2:
                    return "###"
                return "####"
            return "##"

        normalized = str(section_number).strip().rstrip('.')
        if not normalized:
            if section_level is not None:
                if section_level <= 1:
                    return "##"
                if section_level == 2:
                    return "###"
                return "####"
            return "##"

        depth = normalized.count('.') + 1
        if depth <= 1:
            return "##"
        if depth == 2:
            return "###"
        return "####"
    
    def _save_outline_to_file(self):
        """将大纲写入文件开头，格式已在解析时清理"""
        try:
            # 读取现有内容
            with open(self._output_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # 重写文件：标题 + 大纲 + 原有内容
            with open(self._output_file, 'w', encoding='utf-8') as f:
                f.write("# 生成的长文本报告\n\n")
                f.write("## 📋 文章大纲\n\n")
                f.write(self._current_outline)
                f.write("\n\n---\n\n")
                # 只保留原有内容中标题之后的部分
                if "# 生成的长文本报告" in existing_content:
                    parts = existing_content.split("# 生成的长文本报告", 1)
                    if len(parts) > 1:
                        remaining = parts[1].strip()
                        if remaining:
                            f.write(remaining)
                            f.write("\n\n")
            
            self.logger.log(f"✅ 大纲已保存到文件开头: {self._output_file}", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log(f"⚠️ 保存大纲失败: {e}", level=LogLevel.ERROR)
    

    def _log_reflection_to_file(self, stage: str, score: int, report: str):
        """将反思评审结果写入日志文件"""
        try:
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"【{stage}】 - {timestamp}\n")
                f.write(f"评分: {score}\n")
                f.write(f"{'-'*80}\n")
                f.write(report)
                f.write(f"\n{'='*80}\n\n")
        except Exception as e:
            self.logger.log(f"⚠️ 写入反思日志失败: {e}", level=LogLevel.ERROR)
    
    def _log_revision_to_file(self, stage: str, content: str):
        """将修订后的内容写入日志文件"""
        try:
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"【{stage} - 修订结果】 - {timestamp}\n")
                f.write(f"{'-'*80}\n")
                f.write(content[:1000])  # 只记录前1000字符，避免日志文件过大
                if len(content) > 1000:
                    f.write(f"\n... (总长{len(content)}字，已截断)")
                f.write(f"\n{'='*80}\n\n")
        except Exception as e:
            self.logger.log(f"⚠️ 写入修订日志失败: {e}", level=LogLevel.ERROR)

    
    def _filter_references_from_content(self, content: str) -> str:
        """从正文内容中过滤掉参考文献部分
        
        Args:
            content: 原始内容
            
        Returns:
            过滤后的内容（不包含参考文献）
        """
        import re
        
        # 匹配常见的参考文献标题模式
        reference_patterns = [
            r'\n#+\s*参考文献.*',
            r'\n#+\s*References.*',
            r'\n#+\s*引用.*',
            r'\n#+\s*Bibliography.*',
            r'\n\*\*参考文献\*\*.*',
            r'\n\*\*References\*\*.*',
        ]
        
        filtered_content = content
        for pattern in reference_patterns:
            # 找到参考文献标题后，截断内容
            match = re.search(pattern, filtered_content, re.IGNORECASE | re.DOTALL)
            if match:
                # 只保留参考文献标题之前的内容
                filtered_content = filtered_content[:match.start()]
                self.logger.log(f"🔍 检测到参考文献部分，已从正文中移除", level=LogLevel.DEBUG)
                break
        
        return filtered_content.strip()

    def _clean_task_description(self) -> str:
        """清理任务描述，移除框架自动添加的系统提示
        
        框架会在 self.task 中自动添加类似以下的内容：
        - "You're a helpful agent named..."
        - "Your final_answer WILL HAVE to contain these parts..."
        - "### 1. Task outcome (short version):"
        
        此方法提取纯净的用户任务描述。
        
        Returns:
            清理后的任务描述
        """
        task = self.task
        
        # 查找 "Task:" 标记，提取其后的实际任务内容
        task_marker = "Task:\n"
        if task_marker in task:
            # 提取 Task: 后的内容
            task_start = task.find(task_marker) + len(task_marker)
            task_content = task[task_start:]
            
            # 查找分隔符 "---"，这通常标记任务描述的结束
            separator = "\n---\n"
            if separator in task_content:
                task_content = task_content.split(separator)[0]
            
            return task_content.strip()
        
        # 如果没有找到标记，返回原始任务
        return task

    def _preview_for_log(self, value: Any, max_len: Optional[int] = None) -> str:
        """将任意对象转换为可读、可截断的日志预览文本。"""
        limit = max_len or self.max_skill_log_chars
        try:
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                text = str(value)
        except Exception:
            text = repr(value)

        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... (已截断，原始长度 {len(text)} 字符)"

    def _log_skill_call(
        self,
        tool_name: str,
        arguments: Any,
        output: Any = None,
        status: str = "success",
        elapsed_ms: Optional[int] = None,
        error_message: str = "",
    ) -> None:
        """结构化记录Skill调用日志（控制台 + 文件）。"""
        try:
            input_preview = self._preview_for_log(arguments)
            output_preview = self._preview_for_log(output) if output is not None else ""

            # 控制台简要日志
            cost = f" | {elapsed_ms}ms" if elapsed_ms is not None else ""
            if status == "success":
                self.logger.log(f"🧩 Skill调用: {tool_name}{cost}", level=LogLevel.INFO)
                self.logger.log(f"   输入: {input_preview[:240]}", level=LogLevel.DEBUG)
                self.logger.log(f"   输出: {output_preview[:240]}", level=LogLevel.DEBUG)
            else:
                self.logger.log(f"❌ Skill调用失败: {tool_name}{cost}", level=LogLevel.ERROR)
                self.logger.log(f"   输入: {input_preview[:240]}", level=LogLevel.DEBUG)
                self.logger.log(f"   错误: {error_message}", level=LogLevel.ERROR)

            # 文件详细结构化日志（完整内容不截断）
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 将完整内容转换为字符串
            import json
            try:
                full_input = json.dumps(arguments, ensure_ascii=False, indent=2) if isinstance(arguments, (dict, list)) else str(arguments)
            except:
                full_input = str(arguments)
            
            try:
                full_output = json.dumps(output, ensure_ascii=False, indent=2) if isinstance(output, (dict, list)) else str(output) if output is not None else ""
            except:
                full_output = str(output) if output is not None else ""
            
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"【Skill调用】{timestamp}\n")
                f.write(f"current_step: {getattr(self, '_current_step', 'unknown')}\n")
                f.write(f"tool_name: {tool_name}\n")
                f.write(f"status: {status}\n")
                if elapsed_ms is not None:
                    f.write(f"elapsed_ms: {elapsed_ms}\n")
                f.write(f"{'-'*80}\n")
                f.write("[input]\n")
                f.write(full_input + "\n")
                if status == "success":
                    f.write(f"{'-'*80}\n")
                    f.write("[output]\n")
                    f.write(full_output + "\n")
                else:
                    f.write(f"{'-'*80}\n")
                    f.write("[error]\n")
                    f.write(error_message + "\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            self.logger.log(f"⚠️ Skill日志写入失败: {e}", level=LogLevel.ERROR)

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """统一拦截并结构化记录所有Skill调用。"""
        start = time.perf_counter()
        try:
            result = super().execute_tool_call(tool_name, arguments)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._log_skill_call(
                tool_name=tool_name,
                arguments=arguments,
                output=result,
                status="success",
                elapsed_ms=elapsed_ms,
            )
            return result
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._log_skill_call(
                tool_name=tool_name,
                arguments=arguments,
                status="failed",
                elapsed_ms=elapsed_ms,
                error_message=str(e),
            )
            raise
    
    def _step_stream(self, memory_step: ActionStep) -> Generator[ToolOutput | ActionOutput]:
        """
        重写 _step_stream：多阶段长文本生成
        
        进入 LongWriterAgent 的任务一律视为写作任务，
        不再回退到通用 ReAct 流程。
        """
        try:
            self.logger.log_rule("📋 长文本生成流程启动", level=LogLevel.INFO)
            
            # 阶段 1：规划
            self._current_step = "planner"
            self._planning_phase(memory_step)
            
            # 阶段 2：写作
            self._current_step = "writer"
            self._writing_phase(memory_step)
            
            # 阶段 3：整合
            self._current_step = "finalizer"
            self._finalization_phase(memory_step)
            
            # 返回最终答案
            final_output = self._generate_final_output()
            yield ActionOutput(output=final_output, is_final_answer=True)
        
        except Exception as e:
            self.logger.log(f"❌ LongWriterAgent 错误: {e}", level=LogLevel.ERROR)
            raise
    
    def _classify_section_type(self, title: str, index: int, total: int) -> str:
        """识别段落类型：introduction, body, conclusion, references"""
        title_lower = title.lower()

        # 摘要
        if any(kw in title_lower for kw in ["摘要", "abstract", "executive summary"]):
            return "abstract"
        
        # 参考文献
        if any(kw in title_lower for kw in ["参考文献", "references", "bibliography"]):
            return "references"
        
        # 结论
        if any(kw in title_lower for kw in ["结论", "总结", "conclusion", "summary", "展望", "future"]):
            return "conclusion"
        
        # 引言（首个非参考文献段落）
        if index == 0 or any(kw in title_lower for kw in ["引言", "介绍", "introduction", "背景", "background"]):
            return "introduction"
        
        # 普通段落
        return "body"

    def _log_planning_outline_summary(self, outline_v1: str, final_outline: str) -> None:
        """集中记录规划阶段的大纲生成与预览日志。"""
        self.logger.log("📋 初始大纲生成完成", level=LogLevel.INFO)
        self.logger.log(f"大纲预览（前800字）:\n{outline_v1[:800]}...", level=LogLevel.DEBUG)
        self.logger.log("✅ 大纲完成", level=LogLevel.INFO)
        self.logger.log(f"最终大纲预览（前800字）:\n{final_outline[:800]}...", level=LogLevel.DEBUG)

    def _log_section_start(
        self,
        current_section_num: int,
        total_sections: int,
        section_title: str,
        section_type: str,
    ) -> None:
        """集中记录章节开始日志。"""
        self.logger.log(
            f"📝 [{current_section_num}/{total_sections}] {section_title} [{section_type}]",
            level=LogLevel.INFO,
        )

    def _log_section_special_handling(self, message: str) -> None:
        """记录章节特殊处理分支，如延后摘要、跳过非叶子节点。"""
        self.logger.log(message, level=LogLevel.INFO)

    def _log_abstract_materialized(self, at_stage: str) -> None:
        """记录摘要在何时被生成并插入。"""
        self.logger.log(
            f"✅ 摘要已在{at_stage}生成，并已插入到引言之前（内存顺序）",
            level=LogLevel.INFO,
        )

    def _log_section_result(
        self,
        current_section_num: int,
        total_sections: int,
        section_title: str,
        content: str,
        error: Optional[Exception] = None,
    ) -> None:
        """集中记录章节完成或失败日志。"""
        if error is not None:
            self.logger.log(
                f"⚠️ [{current_section_num}/{total_sections}] {section_title} 失败: {error}",
                level=LogLevel.ERROR,
            )
            return

        section_len = len(str(content))
        self.logger.log(
            f"✅ [{current_section_num}/{total_sections}] {section_title} 完成 ({section_len} 字)",
            level=LogLevel.INFO,
        )

    def _log_finalization_summary(self, total_sections: int, total_chars: int) -> None:
        """集中记录整合阶段统计日志。"""
        self.logger.log("📊 生成统计:", level=LogLevel.INFO)
        self.logger.log(f"   总章节数: {total_sections}", level=LogLevel.INFO)
        self.logger.log(f"   总字数: {total_chars}", level=LogLevel.INFO)
        self.logger.log(f"   输出文件: {self._output_file}", level=LogLevel.INFO)
    
    # ============== 阶段 1：规划 ==============
    
    def _planning_phase(self, memory_step: ActionStep) -> None:
        """新流程：基于关键词的5步精准检索 → 综合规划
        
        流程说明：
        1. 概念拆解：将研究主题拆解为2-3个概念组，每组包含3-5个关键词
        2. First-Shot检索：使用初始概念组进行第一次检索，获取Top 20文献
        3. 术语提取：从文献摘要中提取高频专业术语
        4. 概念升级：将高频术语归类到对应概念组，升级检索式
        5. Second-Shot检索：使用升级后的概念组进行精准检索
        6. 大纲生成：基于精准检索结果生成大纲
        """
        self.logger.log_rule("📋 Planning Phase (基于关键词的5步流程)", level=LogLevel.INFO)
        
        try:
            # 清理任务描述，移除框架系统提示
            cleaned_task = self._clean_task_description()
            search_stage_result = self._invoke_workflow_component(
                "keyword_search_expansion",
                {"task": cleaned_task},
            )

            top_20_papers = search_stage_result.get("top_20_papers", [])
            candidate_keywords = search_stage_result.get("candidate_keywords", [])
            upgraded_concepts = search_stage_result.get("upgraded_concepts", [])
            final_retrieval_results = str(search_stage_result.get("final_retrieval_results", ""))

            # ============== 第6-7步：大纲生成 + 反思修订（组件） ==============
            self.logger.log("📝 基于精准检索结果生成大纲...", level=LogLevel.INFO)
            self.logger.log("🔄 大纲反思验证...", level=LogLevel.INFO)
            outline_stage_result = self._invoke_workflow_component(
                "outline_generation_reflection",
                {
                    "task": cleaned_task,
                    "upgraded_concepts": upgraded_concepts,
                    "top_20_papers": top_20_papers,
                    "candidate_keywords": candidate_keywords,
                    "final_retrieval_results": final_retrieval_results,
                },
            )
            outline_v1 = str(outline_stage_result.get("outline_v1", ""))
            self._current_outline = str(outline_stage_result.get("final_outline", outline_v1))
            self._log_planning_outline_summary(outline_v1, self._current_outline)
            
            # 将大纲写入文件开头
            self._save_outline_to_file()
        
        except Exception as e:
            self.logger.log(f"⚠️ 规划失败: {e}", level=LogLevel.ERROR)
            raise
    
    # ============== 阶段 2：写作 ==============
    
    def _writing_phase(self, memory_step: ActionStep) -> None:
        """逐段写作 + 多轮证据反思"""
        self.logger.log_rule("✍️ Writing Phase", level=LogLevel.INFO)

        self._prev_body_or_intro_content = ""  # 写作阶段开始时清空

        sections = self._outline_parsing_service.parse_sections(self, self._current_outline)
        total_sections = len(sections)
        self.logger.log(f"解析出 {total_sections} 章", level=LogLevel.INFO)
        deferred_abstract_section: Optional[Dict[str, str]] = None

        for idx, section in enumerate(sections):
            try:
                # 当前段落序号 (1-based)
                current_section_num = idx + 1
                
                # 识别段落类型
                section_type = self._classify_section_type(
                    section['title'], 
                    idx, 
                    len(sections)
                )
                self._log_section_start(current_section_num, total_sections, section['title'], section_type)

                # 摘要章节延后到结论阶段生成（素材更完整）
                if section_type == "abstract":
                    deferred_abstract_section = section
                    self._log_section_special_handling("⏭️ 摘要章节将延后至结论阶段生成")
                    continue

                if not section.get("is_leaf", True):
                    self._log_section_special_handling("⏭️ 非叶子节点仅作为结构标题保留，不展开正文")
                    final = ""
                    self._generated_sections.append({
                        "title": section['title'],
                        "content": final,
                        "type": section_type,
                        "number": section.get("number", ""),
                        "level": section.get("level"),
                    })
                    self._append_section_to_file(
                        section['title'],
                        final,
                        current_section_num,
                        total_sections,
                        section_type,
                        section.get('number', ''),
                        section.get('level')
                    )
                    self._log_section_result(current_section_num, total_sections, section['title'], final)
                    continue

                # 仅在正文章节调用细粒度 RAG，避免额外 token 占用
                fine_rag_context = ""
                available_citations = {}  # 本段落可用的引用
                if section_type == "body":
                    fine_rag_context = self._run_fine_rag_for_section(section)
                    # 从检索结果中提取可用的引用
                    if fine_rag_context:
                        available_citations = self._citation_flow_service.extract_citations_from_rag(self, fine_rag_context)

                # 根据类型选择 skill，并按章节语义传最小必要上下文
                # 获取字数目标（如果有）
                word_count_target = section.get('word_count_target', 0)
                
                cleaned_task = self._clean_task_description()
                if section_type == "references":
                    # 参考文献章节应传递全局可用引用
                    references_payload = {
                        "section": section,
                        "available_citations": self._citations,
                        "task": cleaned_task
                    }
                    self.logger.log(f"[Component输入] references_writing: {json.dumps(references_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    references_result = self._invoke_workflow_component(
                        "references_writing",
                        references_payload,
                    )
                    self.logger.log(f"[Component输出] references_writing: {json.dumps(references_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    final = str(references_result.get("content", ""))
                elif section_type == "introduction":
                    intro_payload = {
                        "section": section,
                        "available_citations": available_citations,
                        "task": cleaned_task,
                        "prev_section_content": self._prev_body_or_intro_content,
                    }
                    self.logger.log(f"[Component输入] introduction_writing: {json.dumps(intro_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    intro_result = self._invoke_workflow_component(
                        "introduction_writing",
                        intro_payload,
                    )
                    self.logger.log(f"[Component输出] introduction_writing: {json.dumps(intro_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    final = str(intro_result.get("content", ""))
                    self._prev_body_or_intro_content = final
                elif section_type == "conclusion":
                    # 直接使用实时维护的 self._full_text_body
                    conclusion_payload = {
                        "section": section,
                        "available_citations": available_citations,
                        "full_text": self._full_text_body,
                        "task": cleaned_task,
                    }
                    self.logger.log(f"[Component输入] conclusion_writing: {json.dumps(conclusion_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    conclusion_result = self._invoke_workflow_component(
                        "conclusion_writing",
                        conclusion_payload,
                    )
                    self.logger.log(f"[Component输出] conclusion_writing: {json.dumps(conclusion_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    final = str(conclusion_result.get("content", ""))
                    if not self._abstract_section:
                        abstract_config = deferred_abstract_section or {"title": "摘要", "goal": "凝练全文核心发现与价值", "word_count_target": 300}
                        # 直接使用实时维护的 self._full_text_body
                        abstract_payload = {
                            "section": abstract_config,
                            "conclusion_text": final,
                            "full_text": self._full_text_body,
                            "task": cleaned_task,
                        }
                        self.logger.log(f"[Component输入] abstract_writing: {json.dumps(abstract_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                        abstract_result = self._invoke_workflow_component(
                            "abstract_writing",
                            abstract_payload,
                        )
                        self.logger.log(f"[Component输出] abstract_writing: {json.dumps(abstract_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                        abstract_content = str(abstract_result.get("content", ""))
                        if abstract_content:
                            self._abstract_section = {
                                "title": abstract_config.get("title", "摘要"),
                                "content": abstract_content,
                                "type": "abstract",
                                "number": abstract_config.get("number", ""),
                                "level": abstract_config.get("level", 1),
                            }
                            self._insert_abstract_before_introduction()
                            self._log_abstract_materialized("结论阶段")
                else:
                    body_payload = {
                        "section": section,
                        "fine_rag_context": fine_rag_context,
                        "available_citations": available_citations,
                        "task": cleaned_task,
                        "prev_section_content": self._prev_body_or_intro_content,
                    }
                    self.logger.log(f"[Component输入] body_writing: {json.dumps(body_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    body_result = self._invoke_workflow_component(
                        "body_writing",
                        body_payload,
                    )
                    self.logger.log(f"[Component输出] body_writing: {json.dumps(body_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    final = str(body_result.get("content", ""))
                    self._prev_body_or_intro_content = final
                
                # 更新内存（参考文献不参与正文记忆）
                if section_type != "references":
                    section_ref = self._get_section_ref(section)
                    final = self._citation_flow_service.enforce_strict_citation_flow_on_section(
                        str(final),
                        section_ref,
                        available_citations,
                    )
                    self._log_reference_usage_in_section(section_ref, str(final))
                    # 添加该段落中的可用引用到全局引用字典
                    self._citation_flow_service.add_citations(self, available_citations, section_ref)
                    self._citation_flow_service.collect_citations_from_text(
                        self,
                        str(final),
                        section_ref,
                        allowed_citation_keys=set(available_citations.keys()),
                        add_new_from_text=self.allow_add_citation_from_generated_text,
                    )
                    self._update_memory(section['title'], final)
                
                self._generated_sections.append({
                    "title": section['title'],
                    "content": final,
                    "type": section_type,
                    "number": section.get("number", ""),
                    "level": section.get("level"),
                })
                
                # 👇 关键：立即保存到文件，防止丢失
                self._append_section_to_file(
                    section['title'], 
                    final, 
                    current_section_num, 
                    total_sections,
                    section_type,
                    section.get('number', ''),
                    section.get('level')
                )
                self._log_section_result(current_section_num, total_sections, section['title'], final)
            
            except Exception as e:
                self._log_section_result(current_section_num, total_sections, section['title'], "", error=e)

        # 如果没有结论章节但有摘要需求，则在写作末尾补写摘要
        if deferred_abstract_section and not self._abstract_section:
            fallback_payload = {
                "section": deferred_abstract_section,
                "conclusion_text": "",
                "full_text": self._full_text_body,
                "task": self._clean_task_description(),
            }
            self.logger.log(f"[Component输入] abstract_writing: {json.dumps(fallback_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
            fallback_result = self._invoke_workflow_component(
                "abstract_writing",
                fallback_payload,
            )
            self.logger.log(f"[Component输出] abstract_writing: {json.dumps(fallback_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
            fallback_abstract = str(fallback_result.get("content", ""))
            if fallback_abstract:
                self._abstract_section = {
                    "title": deferred_abstract_section.get("title", "摘要"),
                    "content": fallback_abstract,
                    "type": "abstract",
                    "number": deferred_abstract_section.get("number", ""),
                    "level": deferred_abstract_section.get("level", 1),
                }
                self._insert_abstract_before_introduction()
                self._log_abstract_materialized("写作末尾")

    def _run_fine_rag_for_section(self, section: Dict[str, str]) -> str:
        """细粒度RAG检索，支持多种模式
        
        模式说明（由 self.fine_rag_mode 控制）：
        - 'web_agent': 调用text_webbrowser_agent进行深度检索（推荐）
        - 'web_search': 使用DuckDuckGo搜索（快速但内容有限）
        - 'disabled': 禁用细粒度RAG
        """
        section_title = section.get('title', '未命名章节')
        section_goal = section.get('goal', '')
        self.state[self.STATE_LATEST_TAGGED_SEARCH_PAYLOAD] = None
        
        # 检查是否禁用细粒度RAG
        if self.fine_rag_mode == 'disabled':
            self.logger.log(f"⏸️ 细粒度RAG已禁用，章节【{section_title}】将基于已有上下文生成", level=LogLevel.DEBUG)
            return ""
        
        # 构建搜索查询
        search_query = f"{section_title} {section_goal}" if section_goal else section_title
        self.logger.log(f"🔍 为章节【{section_title}】执行细粒度RAG（模式: {self.fine_rag_mode}）", level=LogLevel.DEBUG)
        
        # 模式1：调用text_webbrowser_agent进行深度检索（推荐）
        if self.fine_rag_mode == 'web_agent':
            return self._run_fine_rag_with_web_agent(section_title, search_query)
        
        # 模式2：DuckDuckGo搜索模式（快速但内容有限）
        elif self.fine_rag_mode == 'web_search':
            return self._section_writing_service.run_fine_rag_web_search(self, section_title, search_query)
        
        else:
            self.logger.log(f"⚠️ 未知的细粒度RAG模式: {self.fine_rag_mode}", level=LogLevel.INFO)
            return ""
    
    def _run_fine_rag_with_web_agent(self, section_title: str, search_query: str) -> str:
        """调用text_webbrowser_agent进行细粒度RAG检索
        
        text_webbrowser_agent具有完整的网页浏览能力：
        - DuckDuckGo搜索
        - 访问网页并提取文本
        - 页面导航和内容提取
        """
        try:
            self.logger.log(f"📍 调用改进版 text_webbrowser_agent（tagged）进行深度检索...", level=LogLevel.DEBUG)
            tagged_result = self._tagged_search_service.run_tagged_web_agent_search(
                self,
                query=search_query,
                search_topic=section_title,
                phase="writing_fine_rag",
                top_k=self.tagged_search_top_k,
            )

            context_text = str(tagged_result.get("context_text", "") or "").strip()
            if context_text:
                citations = tagged_result.get("citations", {})
                if citations:
                    # 让引用库直接吃到“检索源”抽出的作者/年份/标题
                    self._citation_flow_service.add_citations(self, citations, section_title)

                payload = tagged_result.get("payload", {})
                self.state.setdefault(self.STATE_SECTION_TAGGED_PAYLOAD, {})[section_title] = payload
                self.logger.log(f"✅ tagged 检索返回内容 ({len(context_text)} 字)", level=LogLevel.INFO)
                return self._trim_text(context_text, self.max_fine_rag_chars)

            self.logger.log(f"⚠️ tagged 检索无返回内容，回退为web_search", level=LogLevel.INFO)
            return self._section_writing_service.run_fine_rag_web_search(self, section_title, search_query)

        except Exception as e:
            self.logger.log(f"⚠️ text_webbrowser_agent 调用异常: {e}，回退为web_search", level=LogLevel.INFO)
            return self._section_writing_service.run_fine_rag_web_search(self, section_title, search_query)
    
    def _clean_agent_output(self, output: str) -> str:
        """清理text_webbrowser_agent返回的内容
        
        移除框架自动添加的格式包装
        """
        text = output.strip()

        # 去掉 managed agent 的工作摘要，避免把超长日志带回主写作链路
        if "<summary_of_work>" in text:
            text = re.sub(r"\n?For more detail, find below a summary of this agent's work:\n<summary_of_work>.*?</summary_of_work>", "", text, flags=re.DOTALL).strip()
        
        # 移除 Task outcome 包装
        markers = [
            "### 1. Task outcome (short version):",
            '"### 1. Task outcome (short version)":',
        ]
        
        for marker in markers:
            if marker in text:
                # 尝试提取Task outcome内容
                match = re.search(
                    r"###\s*1\.\s*Task outcome \(short version\):\s*(.*?)(?:\n###\s*2\.|$)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if match:
                    return match.group(1).strip()
        
        return text

    def _insert_abstract_before_introduction(self) -> None:
        """将摘要插入到引言之前，避免重复插入。"""
        if not self._abstract_section:
            return

        # 先移除可能的已有摘要，确保幂等
        self._generated_sections = [s for s in self._generated_sections if s.get("type") != "abstract"]

        intro_index = next((i for i, s in enumerate(self._generated_sections) if s.get("type") == "introduction"), 0)
        self._generated_sections.insert(intro_index, self._abstract_section)

    def _rewrite_output_file_from_sections(self) -> None:
        """按最终章节顺序重写报告文件（用于保证摘要位于引言前）。"""
        try:
            with open(self._output_file, 'w', encoding='utf-8') as f:
                f.write("# 生成的长文本报告\n\n")
                f.write("## 📋 文章大纲\n\n")
                f.write(self._current_outline)
                f.write("\n\n---\n\n")

                total_sections = len(self._generated_sections)
                for i, section in enumerate(self._generated_sections, 1):
                    title = section.get("title", "未命名章节")
                    section_type = section.get("type", "body")
                    section_number = str(section.get("number", "")).strip()
                    section_level = section.get("level")
                    content = str(section.get("content", ""))
                    heading_prefix = self._get_heading_prefix(section_number, section_level)
                    numbered_title = f"{section_number} {title}".strip() if section_number else title
                    f.write(f"{heading_prefix} {numbered_title} [{i}/{total_sections}]\n\n")
                    if section_type != "references":
                        f.write(self._filter_references_from_content(content))
                    else:
                        f.write(content)
                    f.write("\n\n")
        except Exception as e:
            self.logger.log(f"⚠️ 重写报告文件失败: {e}", level=LogLevel.ERROR)

    def _log_reference_event(self, stage: str, section_title: str, detail: str) -> None:
        """写入独立的参考文献追踪日志。"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._reference_trace_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] [{stage}] 章节: {section_title or '未知章节'}\n")
                f.write(detail + "\n\n")
        except Exception as e:
            self.logger.log(f"⚠️ 写入参考文献追踪日志失败: {e}", level=LogLevel.ERROR)

    def _log_reference_usage_in_section(self, section_title: str, text: str) -> None:
        """记录章节中每条引用落在第几句、具体句子内容。"""
        if not text:
            return

        sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])\s+|\n+', str(text)) if s.strip()]
        pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"

        total_usage = 0
        for idx, sentence in enumerate(sentences, 1):
            matches = re.findall(pattern, sentence)
            for author_raw, year_raw in matches:
                citation_key = f"{author_raw.strip()} ({year_raw.strip()})"
                self._log_reference_event(
                    stage="citation_used",
                    section_title=section_title,
                    detail=(
                        f"引用键: {citation_key}\n"
                        f"句子序号: 第{idx}句\n"
                        f"句子内容: {sentence}"
                    ),
                )
                total_usage += 1

        if total_usage == 0:
            self._log_reference_event(
                stage="citation_used",
                section_title=section_title,
                detail="本章节未检测到 APA 文内引用。",
            )

    def _log_reference_addition(self, source: str, section_title: str, citation_key: str, citation_info: Dict[str, Any]) -> None:
        """记录引用入库详情。"""
        try:
            info_text = json.dumps(citation_info, ensure_ascii=False, indent=2)
        except Exception:
            info_text = str(citation_info)

        self._log_reference_event(
            stage="citation_added",
            section_title=section_title,
            detail=(
                f"来源: {source}\n"
                f"入库键: {citation_key}\n"
                f"入库详情:\n{info_text}"
            ),
        )

    def enable_citation_validation_mode(self, enable: bool = True):
        """启用/禁用引用验证模式
        
        Args:
            enable: 是否启用严格的学术引用验证
        """
        self.enable_citation_validation = enable
        if enable:
            self.logger.log(
                "✅ 启用学术引用验证系统\n"
                "流程: Search → Verify → Retrieve → Validate → Add\n"
                "所有引用将经过双源验证、官方BibTeX获取和摘要核实",
                level=LogLevel.INFO
            )

    def _write_references_section(self, section: Dict[str, str]) -> str:
        """参考文献章节：直接输出收集到的所有引用
        
        不调用 Skill 重新生成，而是使用全文扫描已有的引用
        """
        self._log_reference_event(
            stage="references_section_generated",
            section_title=self._get_section_ref(section),
            detail=f"当前引用库条目数: {len(self._citations)}",
        )
        return self._format_references_section()
    
    # ============== 阶段 3：整合 ==============
    
    def _finalization_phase(self, memory_step: ActionStep) -> None:
        """整合最终输出"""
        self.logger.log_rule("🎁 Finalization", level=LogLevel.INFO)
        
        try:
            # 验证流程接入主链路：在最终定稿前执行 5 步引用验证
            if self.enable_citation_validation:
                validation_component_result = self._invoke_workflow_component(
                    "citation_validation",
                    {},
                )
                validation_result = validation_component_result.get("validation_result", {})
                self.state[self.STATE_CITATION_VALIDATION] = validation_result
                self._refresh_references_sections_content()

            if self._abstract_section:
                self._insert_abstract_before_introduction()

            # 统一按最终章节顺序重写文件，确保“摘要在引言前”
            self._rewrite_output_file_from_sections()

            # 保存到 state
            self.state[self.STATE_OUTLINE] = self._current_outline
            self.state[self.STATE_SECTIONS] = self._generated_sections
            
            # 统计和总结
            total_sections = len(self._generated_sections)
            total_chars = sum(len(str(s.get('content', ''))) for s in self._generated_sections)
            self._log_finalization_summary(total_sections, total_chars)
            
            # 添加统计信息到文件
            with open(self._output_file, 'a', encoding='utf-8') as f:
                f.write("---\n\n")
                f.write("## 生成统计\n\n")
                f.write(f"- 总章节数: {total_sections}\n")
                f.write(f"- 总字数: {total_chars}\n")
                f.write(f"- 生成时间: {self.task}\n")
            
            self.logger.log("✅ 整合完成，所有内容已保存", level=LogLevel.INFO)
        
        except Exception as e:
            self.logger.log(f"⚠️ 整合失败: {e}", level=LogLevel.ERROR)
    
    # ============== 工具方法 ==============

    def _get_section_ref(self, section: Dict[str, Any]) -> str:
        """构造带编号的章节标识，用于引用追踪日志。"""
        number = str(section.get("number", "")).strip()
        title = str(section.get("title", "未知章节")).strip() or "未知章节"
        return f"{number} {title}".strip() if number else title
    
    def _update_memory(self, title: str, content: str):
        """更新内存"""
        self._previous_section_content = content
        
        # 压缩摘要
        if self._global_summary:
            combined = self._global_summary + "\n" + content
        else:
            combined = content
        
        sentences = [s.strip() for s in combined.replace("。", "。\n").split("\n") if s.strip()]
        
        if len(sentences) > 3:
            selected = [sentences[0]] + [sentences[i] for i in range(1, len(sentences), max(1, len(sentences)//3))]
            summary = "。".join(selected)
        else:
            summary = combined
        
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."
        
        self._global_summary = summary
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """解析 JSON"""
        try:
            return json.loads(text)
        except:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            return {"score": 75, "need_revision": False, "evidence_satisfied": True}

    def _trim_text(self, text: str, limit: int) -> str:
        """截断文本，避免上下文过长造成 token 浪费。"""
        if not text:
            return ""
        text = str(text).strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."
    
    def _format_references_section(self) -> str:
        """格式化参考文献列表"""
        if not self._citations:
            return "暂无引用文献。"

        citations_pool = self._citations
        if self.enable_citation_validation:
            verified_only = {
                k: v for k, v in self._citations.items()
                if bool(v.get('verified')) or str(v.get('verification_status', '')).lower() == 'verified'
            }
            if verified_only:
                citations_pool = verified_only
            else:
                self.logger.log("⚠️ 验证模式已启用但暂无通过验证的引用，暂回退输出当前引用池", level=LogLevel.INFO)
        
        # 按作者-年份排序（APA 风格）
        sorted_refs = sorted(
            citations_pool.items(),
            key=lambda x: (str(x[1].get('authors', '')).lower(), str(x[1].get('year', '')))
        )
        
        lines = []
        for idx, (_, info) in enumerate(sorted_refs, 1):
            authors = info.get('authors', '未知作者')
            year = info.get('year', 'n.d.')
            source_title = info.get('title', 'Title unavailable')
            
            # APA 近似兜底格式
            lines.append(f"[{idx}] {authors}. ({year}). {source_title}.")
        
        return "\n".join(lines)

    def _refresh_references_sections_content(self) -> None:
        """在验证后刷新内存中的参考文献章节内容。"""
        if not self._generated_sections:
            return

        refreshed = 0
        new_content = self._format_references_section()
        for section in self._generated_sections:
            if section.get('type') == 'references':
                section['content'] = new_content
                refreshed += 1

        if refreshed:
            self.logger.log(f"✅ 已刷新 {refreshed} 个参考文献章节内容（应用验证结果）", level=LogLevel.INFO)
    
    def _generate_final_output(self) -> str:
        """生成最终输出"""
        try:
            # 读取已保存的文件内容
            with open(self._output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            total_chars = len(content)
            sections_count = len(self._generated_sections)
            
            summary = f"""
# 📝 报告生成完成

## 生成信息
- **输出文件**: {self._output_file}
- **总章节数**: {sections_count}
- **总字数**: {total_chars}
- **任务**: {self.task[:100]}...

## 章节列表
"""
            for i, section in enumerate(self._generated_sections, 1):
                section_len = len(str(section.get('content', '')))
                summary += f"{i}. {section['title']} ({section_len} 字)\n"
            
            summary += f"""
## 说明
✅ 所有内容已实时保存到文件中，防止丢失
📂 你可以在上述路径找到完整的报告文件
🔗 报告格式: Markdown (.md)

## 完整内容已保存，以下是部分预览：
---

{content[:2000]}

...（完整内容已保存到文件）
"""
            return summary
        except Exception as e:
            return f"⚠️ 生成最终输出失败: {e}\n但所有内容已保存到: {self._output_file}"
