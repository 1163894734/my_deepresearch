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
    from .citation_validator import (
        CitationValidator,
        build_canonical_citation_key,
        build_available_citation_maps,
        collect_citations_from_text,
        enqueue_unverified_citation,
        extract_citations_from_rag,
        format_allowed_citation_whitelist,
        resolve_allowed_citation_key,
        scan_citations_in_text,
    )
    from .long_writer import (
        JsonWorkflowComponent,
        KeywordSearchPlanningService,
        OutlineParsingService,
        SectionWritingService,
        TaggedSearchService,
        build_workflow_components,
    )
except ImportError:
    # 兼容直接脚本运行场景
    from citation_validator import (
        CitationValidator,
        build_canonical_citation_key,
        build_available_citation_maps,
        collect_citations_from_text,
        enqueue_unverified_citation,
        extract_citations_from_rag,
        format_allowed_citation_whitelist,
        resolve_allowed_citation_key,
        scan_citations_in_text,
    )
    from long_writer import (
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
        """
        主要作用：初始化长文写作代理或引用验证器所需的运行状态、配置、缓存和外部依赖。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - model: 底层语言模型实例，用于执行规划、写作、反思或引用一致性判断。
        - tools (Optional[List]): 代理可调用的工具列表，会在初始化时注册到工作流执行环境。
        - **kwargs: 额外初始化配置，会透传给父类代理或底层构造流程。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 读取初始化参数并补齐默认配置项。
        - 创建运行期状态、缓存、服务对象和输出路径。
        - 为后续规划、写作、检索、验证和收尾阶段建立稳定执行环境。
        """

        super().__init__(model=model, tools=tools or [], **kwargs)
        
        # 流程状态
        self._current_outline = ""
        self._generated_sections: List[Dict[str, str]] = []
        self._abstract_section: Optional[Dict[str, str]] = None
        
        # 内存管理
        self._previous_section_content = ""
        self._prev_body_or_intro_content = ""  # 实时维护前一段正文或引言内容
        self._global_summary = ""
        self._full_text_body = ""  # 实时维护引言+正文+结论全文文本（供结论/摘要使用）
        
        # 文献管理
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._citation_counter = 0
        
        # 学术引用验证系统
        self._citation_validator = CitationValidator(model=model)
        self._unverified_citations: List[Tuple[str, str, str, str]] = []  # (author, year, title, claim)
        self._queued_citation_keys: Set[str] = set()
        self.enable_citation_validation = True  # 默认启用
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
        self.text_webbrowser_agent_name = "custom_search_agent"  # text_webbrowser_agent 的名称
        self.tagged_search_top_k = 8
        self._tagged_retrieval_records: List[Dict[str, Any]] = []
        self._keyword_search_service = KeywordSearchPlanningService()
        self._outline_parsing_service = OutlineParsingService()
        self._section_writing_service = SectionWritingService()
        self._tagged_search_service = TaggedSearchService()
        self._workflow_components = self._build_workflow_components()
        self.state["workflow_component_contracts"] = self.get_workflow_component_contracts()
        
        # 缓存所有skill的完整提示词（用于日志记录）
        self._skill_prompts: Dict[str, str] = {}
        self._cache_skill_prompts()
    
    def set_text_webbrowser_agent_name(self, agent_name: str):
        """
        主要作用：设置文本网页浏览代理名称，便于后续统一调度网页检索代理。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - agent_name (str): 该参数用于承载 `agent_name` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        self.text_webbrowser_agent_name = agent_name
        self.logger.log(f"✅ text_webbrowser_agent 已配置: {agent_name}", level=LogLevel.INFO)

    def _cache_skill_prompts(self) -> None:
        """
        主要作用：预加载并缓存技能提示词，减少重复文件读取。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        from pathlib import Path
        
        # 从skills目录加载所有skill的提示词
        # long_writer_agent_v3.py 在 scripts 目录，skills 在上级目录 open_deep_research
        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.exists():
            self.logger.log(f"⚠️ Skills目录不存在: {skills_dir}", level=LogLevel.WARNING)
            return
        
        count = 0
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    # 提取frontmatter之后的部分（实际提示词）
                    lines = content.splitlines()
                    if lines and lines[0].strip() == "---":
                        # 找到结束的---
                        end_idx = None
                        for i in range(1, len(lines)):
                            if lines[i].strip() == "---":
                                end_idx = i
                                break
                        if end_idx is not None:
                            # 获取frontmatter之后的内容
                            prompt_body = "\n".join(lines[end_idx + 1:]).lstrip()
                            skill_name = skill_dir.name
                            self._skill_prompts[skill_name] = prompt_body
                            count += 1
                except Exception as e:
                    pass  # 忽略解析失败的skill
        
        self.logger.log(f"✅ 已缓存 {count} 个skill的提示词", level=LogLevel.DEBUG)

    def _build_workflow_components(self) -> Dict[str, JsonWorkflowComponent]:
        """
        主要作用：构建工作流组件注册表。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - Dict[str, JsonWorkflowComponent]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取当前阶段最关键的上下文字段（任务、章节、材料、引用）。
        - 按工作流约定拼装提示词或结构化载荷。
        - 返回可直接交给下游组件执行的输入对象。
        """
        return build_workflow_components()

    def _invoke_workflow_component(self, component_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        主要作用：调用指定工作流组件并返回结构化结果。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - component_name (str): 工作流组件名称。
        - payload (Dict[str, Any]): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        component = self._workflow_components.get(component_name)
        if component is None:
            raise ValueError(f"Unknown workflow component: {component_name}")
        return component.run(self, payload)

    def get_workflow_component_contracts(self) -> Dict[str, Dict[str, Any]]:
        """
        主要作用：收集全部组件的输入输出契约。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - Dict[str, Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        return {
            name: component.contract()
            for name, component in self._workflow_components.items()
        }
    
    def _ensure_output_dir(self):
        """
        主要作用：确保输出目录与相关日志文件路径存在。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
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
        """
        主要作用：将单个章节内容追加写入主输出文件。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - content (str): 正文、章节或日志等具体文本主体。
        - section_index (int): 该参数用于承载 `section_index` 相关的业务上下文或控制信息。
        - total_sections (int): 该参数用于承载 `total_sections` 相关的业务上下文或控制信息。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。
        - section_number (str): 章节编号。
        - section_level (Optional[int]): 章节层级。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
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
        """
        主要作用：根据章节编号和层级生成 Markdown 标题前缀。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section_number (str): 章节编号。
        - section_level (Optional[int]): 章节层级。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
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
        """
        主要作用：将当前大纲写入输出文件。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：把反思阶段的评分与报告写入日志。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - stage (str): 当前流程阶段或日志事件名称。
        - score (int): 评估或反思阶段返回的数值分数。
        - report (str): 模型生成的反思报告、评估报告或日志说明文本。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：把修订后的内容写入日志文件。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - stage (str): 当前流程阶段或日志事件名称。
        - content (str): 正文、章节或日志等具体文本主体。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：从正文中剔除误生成的参考文献段落。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - content (str): 正文、章节或日志等具体文本主体。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
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
        """
        主要作用：清洗任务描述，去掉噪声并保留核心写作目标。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
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
        """
        主要作用：生成适合日志展示的截断预览。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - value (Any): 待格式化或待输出的任意值。
        - max_len (Optional[int]): 该参数用于承载 `max_len` 相关的业务上下文或控制信息。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：记录技能调用的输入、输出和阶段标签。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - tool_name (str): 待调用工具、技能或组件的名称。
        - arguments (Any): 传给工具、技能或组件的参数字典。
        - output (Any): 模型、技能或组件生成的输出文本。
        - status (str): 该参数用于承载 `status` 相关的业务上下文或控制信息。
        - elapsed_ms (Optional[int]): 该参数用于承载 `elapsed_ms` 相关的业务上下文或控制信息。
        - error_message (str): 该参数用于承载 `error_message` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        try:
            input_preview = self._preview_for_log(arguments)
            output_preview = self._preview_for_log(output) if output is not None else ""
            
            # 获取skill的完整提示词（从缓存中查询）
            skill_prompt = self._skill_prompts.get(tool_name, "")
            
            # 提取arguments中的实际输入值
            actual_input = ""
            if isinstance(arguments, dict):
                if "input" in arguments:
                    actual_input = str(arguments.get("input", ""))
                elif "query" in arguments:
                    actual_input = str(arguments.get("query", ""))
                elif "task" in arguments:
                    actual_input = str(arguments.get("task", ""))

            # 控制台简要日志
            cost = f" | {elapsed_ms}ms" if elapsed_ms is not None else ""
            if status == "success":
                self.logger.log(f"🧩 Skill调用: {tool_name}{cost}", level=LogLevel.INFO)
                if skill_prompt:
                    self.logger.log(f"   📋 Skill提示词:\n{skill_prompt}", level=LogLevel.INFO)
                else:
                    self.logger.log(f"   ⚠️ 未找到skill提示词（缓存中有{len(self._skill_prompts)}个skill）", level=LogLevel.DEBUG)
                if actual_input:
                    self.logger.log(f"   📥 用户输入:\n{actual_input}", level=LogLevel.INFO)
                self.logger.log(f"   输入: {input_preview[:240]}", level=LogLevel.DEBUG)
                self.logger.log(f"   输出: {output_preview[:240]}", level=LogLevel.DEBUG)
            else:
                self.logger.log(f"❌ Skill调用失败: {tool_name}{cost}", level=LogLevel.ERROR)
                if skill_prompt:
                    self.logger.log(f"   📋 Skill提示词:\n{skill_prompt}", level=LogLevel.ERROR)
                else:
                    self.logger.log(f"   ⚠️ 未找到skill提示词（缓存中有{len(self._skill_prompts)}个skill）", level=LogLevel.DEBUG)
                if actual_input:
                    self.logger.log(f"   📥 用户输入:\n{actual_input}", level=LogLevel.ERROR)
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
                # 添加skill_prompt诊断信息
                f.write(f"[skill_prompt_diagnosis]\n")
                f.write(f"  缓存中共有 {len(self._skill_prompts)} 个skill\n")
                f.write(f"  查询的tool_name: '{tool_name}'\n")
                f.write(f"  缓存的skill列表: {list(self._skill_prompts.keys())}\n")
                if skill_prompt:
                    f.write(f"  查询结果: ✅ 找到\n")
                else:
                    f.write(f"  查询结果: ❌ 未找到\n")
                f.write(f"{'-'*80}\n")
                if skill_prompt:
                    f.write("[skill_prompt]\n")
                    f.write(skill_prompt + "\n")
                    f.write(f"{'-'*80}\n")
                if actual_input:
                    f.write("[user_input]\n")
                    f.write(actual_input + "\n")
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
        """
        主要作用：统一执行技能、工具或工作流组件调用。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - tool_name (str): 待调用工具、技能或组件的名称。
        - arguments (Dict[str, Any]): 传给工具、技能或组件的参数字典。

        返回值：
        - Any：返回该方法的主要输出结果。

        实现逻辑：
        - 在统一入口接收组件/技能调用，确保所有调用链都经过同一层调度。
        - 统计耗时并记录输入输出快照，形成可回放的运行轨迹。
        - 在异常场景写入失败日志后继续抛错，保证上层流程能做显式失败处理。
        """
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
        主要作用：以流式方式驱动代理的规划、写作和收尾阶段。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - memory_step (ActionStep): 当前 ActionStep，用于驱动代理阶段执行和流式输出。

        返回值：
        - Generator[ToolOutput | ActionOutput]：返回生成器，按阶段流式产出执行结果。

        实现逻辑：
        - 按 planner → writer → finalizer 的顺序串行驱动整条长文生成流水线。
        - 每个阶段内部负责产出下一阶段所需的状态和中间物料，避免阶段间状态漂移。
        - 全流程完成后只输出一个最终答案；任一阶段异常则记录并中断，防止带病结果继续外溢。
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
        """
        主要作用：根据标题和位置判断章节类型。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - title (str): 标题线索，可指论文标题、章节标题或搜索提示中的文献标题。
        - index (int): 当前章节在大纲中的位置索引，用于识别“首段默认引言”等规则。
        - total (int): 大纲总章节数，用于辅助边界判断和后续阶段日志展示。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 优先基于标题关键词识别摘要、参考文献、结论等强语义章节。
        - 对无法直接命中的章节，结合位置规则（如首段默认引言）做兜底判定。
        - 将剩余章节统一归为 body，确保后续组件路由稳定可预测。
        """
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
        """
        主要作用：记录大纲初稿与终稿的差异摘要。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - outline_v1 (str): 大纲初稿，用于和最终大纲做差异对比。
        - final_outline (str): 经过反思修订后的最终大纲文本。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：记录章节生成开始时的关键上下文。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - current_section_num (int): 该参数用于承载 `current_section_num` 相关的业务上下文或控制信息。
        - total_sections (int): 该参数用于承载 `total_sections` 相关的业务上下文或控制信息。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        self.logger.log(
            f"📝 [{current_section_num}/{total_sections}] {section_title} [{section_type}]",
            level=LogLevel.INFO,
        )

    def _log_section_special_handling(self, message: str) -> None:
        """
        主要作用：记录章节写作过程中的特殊处理逻辑。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - message (str): 待解析的输入消息文本，通常来自组件 CLI、模型输出或结构化协议消息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        self.logger.log(message, level=LogLevel.INFO)

    def _log_abstract_materialized(self, at_stage: str) -> None:
        """
        主要作用：记录摘要被实际生成并插入全文的时机。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - at_stage (str): 该参数用于承载 `at_stage` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：记录章节生成结果的摘要信息。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - current_section_num (int): 该参数用于承载 `current_section_num` 相关的业务上下文或控制信息。
        - total_sections (int): 该参数用于承载 `total_sections` 相关的业务上下文或控制信息。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - content (str): 正文、章节或日志等具体文本主体。
        - error (Optional[Exception]): 异常信息或错误文本。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：记录最终整合阶段的统计总结。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - total_sections (int): 该参数用于承载 `total_sections` 相关的业务上下文或控制信息。
        - total_chars (int): 该参数用于承载 `total_chars` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        self.logger.log("📊 生成统计:", level=LogLevel.INFO)
        self.logger.log(f"   总章节数: {total_sections}", level=LogLevel.INFO)
        self.logger.log(f"   总字数: {total_chars}", level=LogLevel.INFO)
        self.logger.log(f"   输出文件: {self._output_file}", level=LogLevel.INFO)
    
    # ============== 阶段 1：规划 ==============
    
    def _planning_phase(self, memory_step: ActionStep) -> None:
        """
        主要作用：执行规划阶段，完成概念提取、检索扩展和大纲生成。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - memory_step (ActionStep): 当前 ActionStep，用于驱动代理阶段执行和流式输出。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 按顺序驱动该阶段的子步骤执行。
        - 同步更新代理状态、日志和落盘文件。
        - 为下一阶段提供一致、完整的中间结果。
        """
        self.logger.log_rule("📋 Planning Phase (基于关键词的5步流程)", level=LogLevel.INFO)
        
        try:
            # 清理任务描述，移除框架系统提示
            cleaned_task = self._clean_task_description()
            search_stage_result = self._invoke_workflow_component(
                "keyword_search_expansion",
                {"task": cleaned_task},
            )
            
            available_citations = search_stage_result.get("available_citations", [])

            # ============== 第6-7步：大纲生成 + 反思修订（组件） ==============
            self.logger.log("📝 基于精准检索结果生成大纲...", level=LogLevel.INFO)
            self.logger.log("🔄 大纲反思验证...", level=LogLevel.INFO)
            outline_stage_result = self._invoke_workflow_component(
                "outline_generation_reflection",
                {
                    "task": cleaned_task,
                    "available_citations": available_citations,
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
        """
        主要作用：执行写作阶段，逐章节生成内容并做引用治理。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - memory_step (ActionStep): 当前 ActionStep，用于驱动代理阶段执行和流式输出。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 按顺序驱动该阶段的子步骤执行。
        - 同步更新代理状态、日志和落盘文件。
        - 为下一阶段提供一致、完整的中间结果。
        """
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
                        available_citations = json.loads(fine_rag_context)

                # 根据类型选择 skill，并按章节语义传最小必要上下文
                
                cleaned_task = self._clean_task_description()
                if section_type == "references":
                    # 参考文献章节应传递全局可用引用
                    references_available_citations = self._citation
                    references_payload = {
                        "section": section,
                        "available_citations": references_available_citations,
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
                    available_citations, final = self._citation_validator.run_five_step_validation(available_citations,final)
                    self._prev_body_or_intro_content = final
                    self._citations={**self._citations,**available_citations}
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
                    available_citations, final = self._citation_validator.run_five_step_validation(available_citations,final)
                    self._prev_body_or_intro_content = final
                    self._citations={**self._citations,**available_citations}
                elif section_type == "abstract":
                    # 直接使用实时维护的 self._full_text_body
                    abstract_payload = {
                        "section": section,
                        "full_text": self._full_text_body,
                        "task": cleaned_task,
                    }
                    self.logger.log(f"[Component输入] abstract: {json.dumps(abstract_payload, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    abstract_result = self._invoke_workflow_component(
                        "abstract_writing",
                        abstract_payload,
                    )
                    self.logger.log(f"[Component输出] abstract: {json.dumps(abstract_result, ensure_ascii=False, indent=2)}", level=LogLevel.INFO)
                    final = str(abstract_result.get("content", ""))                 
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
                    available_citations, final = self._citation_validator.run_five_step_validation(available_citations,final)
                    self._prev_body_or_intro_content = final
                    self._citations={**self._citations,**available_citations}
                
                # 更新内存（参考文献不参与正文记忆）
                if section_type != "references":
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
                raise

    def _run_fine_rag_for_section(self, section: Dict[str, str]) -> str:
        """
        主要作用：为单个章节运行细粒度检索增强流程。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
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
            raise ValueError(f"未知的细粒度RAG模式: {self.fine_rag_mode}")
    
    def _run_fine_rag_with_web_agent(self, section_title: str, search_query: str) -> str:
        """
        主要作用：调用网页检索代理构造章节级细粒度 RAG 材料。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
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
            self.logger.log(f"📍 调用改进版 text_webbrowser_agent（tagged）进行深度检索...", level=LogLevel.DEBUG)
            tagged_result = self._tagged_search_service.run_tagged_web_agent_search(
                self,
                query=search_query,
                search_time="2024-2026",
                top_k=self.tagged_search_top_k,
            )
            return json.dumps(tagged_result.get("available_citations"))
        except Exception as e:
            self.logger.log(f"⚠️ text_webbrowser_agent 调用异常: {e}", level=LogLevel.ERROR)
            raise
    
    def _clean_agent_output(self, output: str) -> str:
        """
        主要作用：清洗代理输出中的包装文本和无关标记。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - output (str): 模型、技能或组件生成的输出文本。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
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
        """
        主要作用：将摘要章节插入到引言前。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        if not self._abstract_section:
            return

        # 先移除可能的已有摘要，确保幂等
        self._generated_sections = [s for s in self._generated_sections if s.get("type") != "abstract"]

        intro_index = next((i for i, s in enumerate(self._generated_sections) if s.get("type") == "introduction"), 0)
        self._generated_sections.insert(intro_index, self._abstract_section)

    def _rewrite_output_file_from_sections(self) -> None:
        """
        主要作用：根据当前章节状态重写主输出文件。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：写入引用治理的详细追踪事件。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - stage (str): 当前流程阶段或日志事件名称。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - detail (str): 该参数用于承载 `detail` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._reference_trace_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] [{stage}] 章节: {section_title or '未知章节'}\n")
                f.write(detail + "\n\n")
        except Exception as e:
            self.logger.log(f"⚠️ 写入参考文献追踪日志失败: {e}", level=LogLevel.ERROR)

    def _log_reference_usage_in_section(self, section_title: str, text: str) -> None:
        """
        主要作用：扫描并记录某章节实际使用的文内引用。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - text (str): 待解析、清洗或重写的原始文本内容。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：记录引用条目被新增到引用库的过程。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - source (str): 数据来源标识，例如 crossref、arxiv、openalex、rag_context 或 generated_text。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - citation_key (str): 引用库内部使用的键，通常是 canonical_key。
        - citation_info (Dict[str, Any]): 某条引用的元数据字典。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
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
        """
        主要作用：开启或关闭五步引用验证模式。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - enable (bool): 该参数用于承载 `enable` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
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
        """
        主要作用：生成参考文献章节内容。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section (Dict[str, str]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        self._log_reference_event(
            stage="references_section_generated",
            section_title=self._get_section_ref(section),
            detail=f"当前引用库条目数: {len(self._citations)}",
        )
        return self._format_references_section()

    def _append_citation_validation_text_log(
        self,
        section_type: str,
        section_ref: str,
        before_text: str,
        after_text: str,
        validation_result: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> None:
        """
        主要作用：将引用验证前后文本和验证结果写入 JSON 日志。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。
        - section_ref (str): 章节引用名，通常用于日志、引用验证和输出文件中的稳定标识。
        - before_text (str): 引用验证或修订前的章节文本。
        - after_text (str): 引用验证或修订后的章节文本。
        - validation_result (Optional[Dict[str, Any]]): 该参数用于承载 `validation_result` 相关的业务上下文或控制信息。
        - error (str): 异常信息或错误文本。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """
        try:
            existing_data: Dict[str, Any] = {}
            try:
                with open(self._citations_validation_log, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_data = loaded
            except Exception:
                existing_data = {}

            section_text_logs = existing_data.get("section_text_logs", [])
            if not isinstance(section_text_logs, list):
                section_text_logs = []

            summary: Dict[str, Any] = {}
            if isinstance(validation_result, dict):
                summary = {
                    "status": validation_result.get("status"),
                    "total": validation_result.get("total"),
                    "verified": validation_result.get("verified"),
                    "rejected": validation_result.get("rejected"),
                    "verification_rate": validation_result.get("verification_rate"),
                }

            section_text_logs.append(
                {
                    "timestamp": time.time(),
                    "section_type": section_type,
                    "section_title": section_ref,
                    "before_text": str(before_text or ""),
                    "after_text": str(after_text or ""),
                    "changed": str(before_text or "") != str(after_text or ""),
                    "validation_result_summary": summary,
                    "error": str(error or ""),
                }
            )

            existing_data["section_text_logs"] = section_text_logs

            with open(self._citations_validation_log, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.log(f"⚠️ 写入引用验证前后文本日志失败: {e}", level=LogLevel.ERROR)
            raise

    def _run_inline_citation_validation_for_section(
        self,
        section_type: str,
        section_ref: str,
        section_content: str,
        available_citations: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        主要作用：对单个章节执行段落级即时引用验证，并把验证结果反向作用到正文内容。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section_type (str): 章节类型标识，如 body、introduction、conclusion、abstract 或 references。
        - section_ref (str): 章节引用名，通常用于日志、引用验证和输出文件中的稳定标识。
        - section_content (str): 某个章节当前版本的正文内容。
        - available_citations (Optional[Dict[str, Dict[str, Any]]]): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 先把本段候选引用按 author/year 入待验证队列，再调用五步验证组件批处理。
        - 仅保留验证通过的引用进入全局引用库，并将失败引用标记为 rejected。
        - 对失败引用执行“删整句”清理，再做严格白名单重写，确保最终正文不含无依据引用断言。
        """
        current = str(section_content or "")
        before_validation_text = str(current)
        if section_type not in {"body", "introduction", "conclusion"}:
            return current
        if not self.enable_citation_validation:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"【引用验证】{timestamp}\n")
                    f.write(f"section_type: {section_type}\n")
                    f.write(f"section_title: {section_ref}\n")
                    f.write("status: skipped\n")
                    f.write("reason: enable_citation_validation=False（未启用五步验证）\n")
                    f.write(f"{'='*80}\n")
            except Exception as e:
                self.logger.log(f"⚠️ 写入引用验证跳过日志失败: {e}", level=LogLevel.ERROR)

            candidate_pool, _candidate_canonical_map = self._citation_flow_service.build_available_citation_maps(available_citations if isinstance(available_citations, dict) else {})
            for title_key, info in candidate_pool.items():
                canonical_key = str(info.get("canonical_key") or self._citation_flow_service.build_canonical_citation_key(info.get("authors", ""), info.get("year", ""), title_key)).strip()
                if not canonical_key:
                    continue
                existing = self._citations.get(canonical_key, {})
                if canonical_key not in self._citations:
                    self._citation_counter += 1
                merged = dict(existing)
                merged.update(dict(info or {}))
                if "id" not in merged:
                    merged["id"] = self._citation_counter
                self._citations[canonical_key] = merged
            return current

        validation_result: Dict[str, Any] = {}
        error_text = ""
        try:
            if available_citations and not isinstance(available_citations, dict):
                raise ValueError("available_citations 必须是 Dict[\"title\", citation_info]，不再接受 list")
            candidate_pool, _candidate_canonical_map = self._citation_flow_service.build_available_citation_maps(available_citations if isinstance(available_citations, dict) else {})

            # 1) 先把当前段落可用引用入待验证队列（不直接入 _citations）
            for key, info in candidate_pool.items():
                author = str(info.get("authors") or "").strip()
                year = str(info.get("year") or "").strip()
                title = str(info.get("title") or "").strip()
                if author and year:
                    self._citation_flow_service.enqueue_unverified_citation(
                        self,
                        author=author,
                        year=year,
                        title=title,
                        claim=f"章节「{section_ref}」候选引用验证",
                    )

            # 2) 执行五步验证
            validation_component_result = self._invoke_workflow_component("citation_validation", {})
            validation_result = validation_component_result.get("validation_result", {})
            history = self.state.get(self.STATE_CITATION_VALIDATION)
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "stage": "per_section",
                    "section_type": section_type,
                    "section_title": section_ref,
                    "result": validation_result,
                }
            )
            self.state[self.STATE_CITATION_VALIDATION] = history

            # 3) 把验证状态写回 available_citations（is_valid）
            results_map = validation_result.get("results", {}) if isinstance(validation_result, dict) else {}
            verified_available: Dict[str, Dict[str, Any]] = {}
            rejected_keys: Set[str] = set()
            for key, info in candidate_pool.items():
                author = str(info.get("authors") or "").strip()
                year = str(info.get("year") or "").strip()
                canonical_key = str(info.get("canonical_key") or self._citation_flow_service.build_canonical_citation_key(author, year, key)).strip()
                result_key = f"{author}_{year}" if author and year else ""
                is_valid = False
                if result_key and isinstance(results_map, dict):
                    row = results_map.get(result_key, {})
                    is_valid = str(row.get("status", "")).lower() == "verified"

                if isinstance(info, dict):
                    info["is_valid"] = is_valid

                if is_valid:
                    verified_available[key] = dict(info or {})
                    verified_available[key]["verified"] = True
                    verified_available[key]["verification_status"] = "verified"
                else:
                    if canonical_key:
                        rejected_keys.add(canonical_key)

            # 4) 仅将验证通过的 available_citations 入库（去重 + 维护 _citation_counter）
            for title_key, info in verified_available.items():
                canonical_key = str(info.get("canonical_key") or self._citation_flow_service.build_canonical_citation_key(info.get("authors", ""), info.get("year", ""), title_key)).strip()
                if not canonical_key:
                    continue
                existing = self._citations.get(canonical_key, {})
                if canonical_key not in self._citations:
                    self._citation_counter += 1

                merged = dict(existing)
                merged.update(dict(info or {}))
                merged["verified"] = True
                merged["verification_status"] = "verified"
                if "id" not in merged:
                    merged["id"] = self._citation_counter
                self._citations[canonical_key] = merged

                if not existing:
                    self._log_reference_addition(
                        source="validated_available_citations",
                        section_title=section_ref,
                        citation_key=canonical_key,
                        citation_info=merged,
                    )

            # 5) 清理：凡验证失败引用，删除其所在整句（不仅删括号）
            if rejected_keys:
                removed_count = 0
                kept_sentences: List[str] = []
                sentences = [s for s in re.split(r'(?<=[。！？!?])\s+|\n+', str(current)) if s and s.strip()]
                for sentence in sentences:
                    findings = self._citation_flow_service.scan_citations_in_text(sentence)
                    hit_keys: List[str] = []
                    for item in findings:
                        canonical_key = self._citation_flow_service.resolve_allowed_citation_key(item.get("raw_key", ""), rejected_keys)
                        if canonical_key and canonical_key in rejected_keys:
                            hit_keys.append(canonical_key)

                    if hit_keys:
                        removed_count += 1
                        self._log_reference_event(
                            stage="citation_sentence_removed",
                            section_title=section_ref,
                            detail=(
                                "因引用验证失败，删除包含该引用的整句\n"
                                f"rejected_keys: {sorted(set(hit_keys))}\n"
                                f"deleted_sentence: {str(sentence).strip()[:500]}"
                            ),
                        )
                    else:
                        kept_sentences.append(sentence.strip())

                current = "\n".join([s for s in kept_sentences if s])
                self._log_reference_event(
                    stage="post_validation_sentence_cleanup",
                    section_title=section_ref,
                    detail=(
                        "验证后句子清理\n"
                        f"rejected_keys_count: {len(rejected_keys)}\n"
                        f"removed_sentence_count: {removed_count}\n"
                        f"rejected_keys: {sorted(rejected_keys)}"
                    ),
                )

            current = str(current)
        except Exception as e:
            error_text = str(e)
            self.logger.log(f"⚠️ 段落级引用验证失败（{section_ref}）: {e}", level=LogLevel.ERROR)
            raise

        self._append_citation_validation_text_log(
            section_type=section_type,
            section_ref=section_ref,
            before_text=before_validation_text,
            after_text=str(current),
            validation_result=validation_result,
            error=error_text,
        )

        # 同步写入 generation_log.txt，方便直接查看主日志时也能看到验证结果
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            vr = validation_result if isinstance(validation_result, dict) else {}
            v_status = vr.get("status", "unknown")
            v_total = vr.get("total", 0)
            v_verified = vr.get("verified", 0)
            v_rejected = vr.get("rejected", 0)
            text_changed = str(before_validation_text) != str(current)
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"【引用验证】{timestamp}\n")
                f.write(f"section_type: {section_type}\n")
                f.write(f"section_title: {section_ref}\n")
                f.write(f"status: {v_status}\n")
                f.write(f"total: {v_total} | verified: {v_verified} | rejected: {v_rejected}\n")
                f.write(f"text_changed: {text_changed}\n")
                if error_text:
                    f.write(f"error: {error_text}\n")
                f.write(f"详细结果 → {self._citations_validation_log}\n")
                f.write(f"{'='*80}\n")
        except Exception as _log_err:
            self.logger.log(f"⚠️ 写入引用验证运行日志失败: {_log_err}", level=LogLevel.ERROR)

        return current
    
    # ============== 阶段 3：整合 ==============
    
    def _finalization_phase(self, memory_step: ActionStep) -> None:
        """
        主要作用：执行最终整合阶段，刷新参考文献并清理未通过验证的引用。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - memory_step (ActionStep): 当前 ActionStep，用于驱动代理阶段执行和流式输出。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 按顺序驱动该阶段的子步骤执行。
        - 同步更新代理状态、日志和落盘文件。
        - 为下一阶段提供一致、完整的中间结果。
        """
        self.logger.log_rule("🎁 Finalization", level=LogLevel.INFO)
        
        try:
            # 段落级验证已在写作阶段执行；最终阶段仅做收口清理。
            if self.enable_citation_validation:
                self._refresh_references_sections_content()
                self._clean_invalid_citations_from_text()

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
            raise
    
    # ============== 工具方法 ==============

    def _get_section_ref(self, section: Dict[str, Any]) -> str:
        """
        主要作用：根据章节字典生成稳定的章节引用名。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - section (Dict[str, Any]): 章节配置字典，通常包含标题、目标、编号、层级和目标字数等字段。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        number = str(section.get("number", "")).strip()
        title = str(section.get("title", "未知章节")).strip() or "未知章节"
        return f"{number} {title}".strip() if number else title
    
    def _update_memory(self, title: str, content: str):
        """
        主要作用：更新代理记忆、全文缓存与状态对象。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - title (str): 标题线索，可指论文标题、章节标题或搜索提示中的文献标题。
        - content (str): 正文、章节或日志等具体文本主体。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        self._previous_section_content = content

        # 实时维护全文正文缓存（供 conclusion/abstract 的 full_text 使用）
        block_title = str(title or "").strip() or "未命名章节"
        block_content = str(content or "").strip()
        if block_content:
            block = f"## {block_title}\n\n{block_content}"
            if self._full_text_body:
                self._full_text_body = f"{self._full_text_body}\n\n{block}"
            else:
                self._full_text_body = block
        
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
        """
        主要作用：尽量稳健地把文本解析成 JSON 字典。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - text (str): 待解析、清洗或重写的原始文本内容。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
        """
        try:
            return json.loads(text)
        except Exception as first_error:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception as second_error:
                    raise ValueError(f"JSON解析失败（代码块内容无效）: {second_error}") from second_error
            raise ValueError(f"JSON解析失败（无有效JSON）: {first_error}") from first_error

    def _trim_text(self, text: str, limit: int) -> str:
        """
        主要作用：将文本裁剪到指定长度范围内。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - limit (int): 文本截断长度或输出上限。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
        """
        if not text:
            return ""
        text = str(text).strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."
    
    def _format_references_section(self) -> str:
        """
        主要作用：将已验证引用库格式化为参考文献段落。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
        if not self._citations:
            return "暂无引用文献。"

        # 直接按 _citations 输出；入库策略已保证只存放需要输出的条目
        citations_pool = self._citations
        
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
        """
        主要作用：刷新所有参考文献章节内容，使其与引用库保持一致。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
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
    def _clean_invalid_citations_from_text(self) -> None:
        """
        主要作用：全局清理正文中无效或未验证的引用。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 先从引用库中抽取最终 verified 白名单，作为正文清理的唯一合法依据。
        - 对每个非参考文献章节逐句扫描文内引用，命中未验证引用则整句删除。
        - 清理后回写章节内容并输出统计日志，保证最终报告与验证结果严格一致。
        """
        if not self._generated_sections or not self.enable_citation_validation:
            return

        # 1. 找出所有“幸存”的合法引用键（白名单）
        verified_keys = {
            k for k, v in self._citations.items()
            if bool(v.get('verified')) or str(v.get('verification_status', '')).lower() == 'verified'
        }

        cleaned_count = 0
        
        # 2. 遍历所有非参考文献章节
        for section in self._generated_sections:
            if section.get('type') == 'references':
                continue
                
            original_text = str(section.get('content', ''))
            if not original_text:
                continue

            import re
            # 按照中文和英文的句号、叹号、问号（包含可能的引号）来切分句子，并保留标点符号
            sentences = re.split(r'([。！？.!?][”"’\']?)', original_text)
            
            new_sentences = []
            section_modified = False
            i = 0
            
            # sentences 的结构是 ['句子1', '标点1', '句子2', '标点2']
            while i < len(sentences):
                sentence_text = sentences[i]
                punctuation = sentences[i+1] if i + 1 < len(sentences) else ""
                full_sentence = sentence_text + punctuation
                
                if not full_sentence.strip():
                    i += 2
                    continue

                # 扫描这句话里是否含有引用
                findings = self._citation_flow_service.scan_citations_in_text(full_sentence)
                keep_sentence = True
                
                if findings:
                    allowed_keys_set = set(self._citations.keys())
                    for item in findings:
                        canonical_key = self._citation_flow_service.resolve_allowed_citation_key(
                            item["raw_key"], allowed_keys_set
                        )
                        # 一旦发现这句话里有未通过验证的引用（假文献/滥用文献）
                        if canonical_key not in verified_keys:
                            keep_sentence = False
                            cleaned_count += 1
                            section_modified = True
                            self.logger.log(f"✂️ 连坐删减：已抹除含无效引用的断言 -> {full_sentence.strip()}", level=LogLevel.DEBUG)
                            break  # 整句干掉，直接跳出不用再看这句话的其他引用了
                
                # 如果这句话清清白白，就保留下来
                if keep_sentence:
                    new_sentences.append(full_sentence)
                
                i += 2
            
            # 3. 将清理后的句子重新拼合成段落
            if section_modified:
                section['content'] = "".join(new_sentences).strip()

        if cleaned_count > 0:
            self.logger.log(f"🧹 扫尾清理：已自动【连句带引】擦除了正文中的 {cleaned_count} 处无支撑断言", level=LogLevel.INFO)
        
    def _generate_final_output(self) -> str:
        """
        主要作用：汇总当前状态并生成最终输出文本。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
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
            raise RuntimeError(f"生成最终输出失败: {e}") from e
