"""
LongWriterAgent V2 - 多阶段长文本生成代理

核心设计：
✅ 继承 ToolCallingAgent 的完整能力（run(), memory, logger, execute_tool_call）
✅ 重写 _step_stream() 实现多阶段流程（规划 → 写作 → 整合）
✅ 多轮反思循环（大纲和每段都支持）
✅ 严格的上下文控制（只保留前一段+摘要）
✅ 内置文献管理（自动编号，避免重复）
✅ 证据达标机制（段落必须满足证据要求才停止）
✅ 学术引用验证系统（5步验证流程：Search→Verify→Retrieve→Validate→Add）

关键：不重新实现 Agent 循环，只自定义 _step_stream() 的处理逻辑
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
import json
import re
import time

from smolagents import ToolCallingAgent
from smolagents.agents import ToolOutput, ActionOutput
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel

# 导入引用验证系统
try:
    from .citation_validator import CitationValidator, CitationRecord
except ImportError:
    # 兼容直接脚本运行场景
    from citation_validator import CitationValidator, CitationRecord


class LongWriterAgent(ToolCallingAgent):
    """
    多阶段长文本写作代理
    
    🎯 职责：
    1. 控制完整的写作流程（规划 → 写作）
    2. 执行多轮反思循环（保证质量）
    3. 管理内存和文献（内部管理）
    4. 调用各个 Skill 完成具体任务
    
    🏗️ 扩展策略：
    - 继承 ToolCallingAgent 的 run(), memory, logger, execute_tool_call
    - 重写 _step_stream() 实现多阶段流程
    - 内部维护 outline, sections, citations
    """
    
    def __init__(self, model, tools: Optional[List] = None, **kwargs):
        super().__init__(model=model, tools=tools or [], **kwargs)
        
        # 流程状态
        self._current_outline = ""
        self._generated_sections: List[Dict[str, str]] = []
        
        # 内存管理
        self._previous_section_content = ""
        self._global_summary = ""
        
        # 文献管理
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._citation_counter = 0
        
        # 学术引用验证系统
        self._citation_validator = CitationValidator(model=model)
        self._unverified_citations: List[Tuple[str, str, str, str]] = []  # (author, year, title, claim)
        self.enable_citation_validation = False  # 默认禁用（API调用较多）
        self.citation_validation_timeout = 5  # API超时时间
        
        # 输出文件管理（实时保存）
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._output_dir = f"outputs/report_{timestamp}"
        self._output_file = f"{self._output_dir}/report.md"
        self._log_file = f"{self._output_dir}/generation_log.txt"
        self._citations_validation_log = f"{self._output_dir}/citations_validation.json"
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
        # 可用模式: 'web_search' (web搜索), 'knowledge' (仅模型知识), 'disabled' (不使用粗细RAG)
        self.coarse_rag_mode = 'web_search'
        
        # 细粒度RAG配置（调用text_webbrowser_agent进行深度检索）
        self.enable_fine_rag_web_search = True  # 是否启用细粒度RAG的web检索
        # 可用模式: 'web_agent' (调用text_webbrowser_agent), 'web_search' (DuckDuckGo搜索), 'disabled' (禁用)
        self.fine_rag_mode = 'web_agent'
        self.text_webbrowser_agent_name = "search_agent"  # text_webbrowser_agent 的名称
    
    def set_text_webbrowser_agent_name(self, agent_name: str):
        """设置text_webbrowser_agent的名称（用于调用managed agent）
        
        Args:
            agent_name: managed agent 的名称，默认为 "search_agent"
        """
        self.text_webbrowser_agent_name = agent_name
        self.logger.log(f"✅ text_webbrowser_agent 已配置: {agent_name}", level=LogLevel.INFO)
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        import os
        os.makedirs(self._output_dir, exist_ok=True)
        # 初始化输出文件
        with open(self._output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 生成的长文本报告\n\n")
            f.write(f"生成时间: {self.task}\n\n")
            f.write("---\n\n")
    
    def _append_section_to_file(self, section_title: str, content: str, section_index: int = 0, total_sections: int = 0, section_type: str = "body"):
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
                f.write(f"## {section_title}{progress}\n\n")
                
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
        
        不是单步的"思考→执行"，而是多阶段的"规划→反思→修正→写作"
        """
        try:
            # 判断是否为写作任务
            if self._is_writing_task(self.task):
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
            else:
                # 非写作任务，用父类的 ReAct 流程
                self._current_step = "react"
                yield from super()._step_stream(memory_step)
        
        except Exception as e:
            self.logger.log(f"❌ LongWriterAgent 错误: {e}", level=LogLevel.ERROR)
            raise
    
    def _is_writing_task(self, task: str) -> bool:
        """判断是否为写作任务"""
        keywords = ["写", "撰写", "论文", "文章", "综述", "报告", "学术"]
        return any(kw in task for kw in keywords)
    
    def _classify_section_type(self, title: str, index: int, total: int) -> str:
        """识别段落类型：introduction, body, conclusion, references"""
        title_lower = title.lower()
        
        # 参考文献
        if any(kw in title_lower for kw in ["参考文献", "references", "bibliography", "引用"]):
            return "references"
        
        # 结论
        if any(kw in title_lower for kw in ["结论", "总结", "conclusion", "summary", "展望", "future"]):
            return "conclusion"
        
        # 引言（首个非参考文献段落）
        if index == 0 or any(kw in title_lower for kw in ["引言", "介绍", "introduction", "背景", "background"]):
            return "introduction"
        
        # 普通段落
        return "body"
    
    # ============== 阶段 1：规划 ==============
    
    def _planning_phase(self, memory_step: ActionStep) -> None:
        """改进的规划：问题分解 → 多角度检索 → 综合规划"""
        self.logger.log_rule("📋 Planning Phase", level=LogLevel.INFO)
        
        try:
            # 清理任务描述，移除框架系统提示
            cleaned_task = self._clean_task_description()
            
            # 1. 问题分解：将主任务分解为多个子问题
            self.logger.log("📍 问题分解...", level=LogLevel.INFO)
            sub_questions = self._decompose_task(cleaned_task)
            self.logger.log(f"✅ 生成 {len(sub_questions)} 个子问题", level=LogLevel.INFO)
            
            # 2. 多角度检索：为每个子问题检索相关材料
            self.logger.log("🔍 多角度检索...", level=LogLevel.INFO)
            retrieval_results = {}
            for i, sub_q in enumerate(sub_questions, 1):
                self.logger.log(f"  检索子问题 {i}/{len(sub_questions)}: {sub_q[:50]}...", level=LogLevel.DEBUG)
                result = self._retrieve_for_subquestion(sub_q)
                retrieval_results[sub_q] = result
            
            # 3. 综合检索结果和子问题，生成大纲
            self.logger.log("📝 综合规划...", level=LogLevel.INFO)
            outline_input = self._build_outline_input(sub_questions, retrieval_results)
            
            outline_v1 = self.execute_tool_call(
                "outline_generation",
                {"input": outline_input}
            )
            outline_v1 = str(outline_v1)
            
            self.logger.log("📋 初始大纲生成完成", level=LogLevel.INFO)
            self.logger.log(f"大纲预览（前800字）:\n{outline_v1[:800]}...", level=LogLevel.DEBUG)
            
            # 4. 多轮反思循环（包含覆盖性校验）
            self.logger.log("🔄 大纲反思验证（含子问题覆盖性检查）...", level=LogLevel.INFO)
            self._current_outline = self._outline_reflection_loop(outline_v1, sub_questions)
            
            self.logger.log("✅ 大纲完成", level=LogLevel.INFO)
            self.logger.log(f"最终大纲预览（前800字）:\n{self._current_outline[:800]}...", level=LogLevel.DEBUG)
            
            # 5. 将大纲写入文件开头
            self._save_outline_to_file()
        
        except Exception as e:
            self.logger.log(f"⚠️ 规划失败: {e}", level=LogLevel.ERROR)
            raise

    def _decompose_task(self, task: str) -> list[str]:
        """将主任务分解为多个子问题，根据 task_decompose Skill 的结果决定数量"""
        try:
            self.logger.log("📍 调用 task_decompose Skill...", level=LogLevel.DEBUG)
            response = self.execute_tool_call("task_decompose", {"input": task})
            response = str(response)
            
            # 解析子问题
            lines = response.strip().split('\n')
            all_questions = []
            for line in lines:
                line = line.strip()
                # 提取格式 "1. ..." 或 "1. [...]" 的问题
                if line and line[0].isdigit():
                    # 移除编号、点和空格
                    q = line.split('. ', 1)[-1].strip()
                    # 移除方括号（如果有）
                    q = q.replace('[', '').replace(']', '').strip()
                    if q and len(q) > 5:  # 确保问题足够长
                        all_questions.append(q)
            
            if not all_questions:
                self.logger.log("⚠️ 未能解析子问题，使用简单分解", level=LogLevel.INFO)
                fallback_questions = self._simple_decompose(task)
                self.logger.log(f"📋 简单分解结果（{len(fallback_questions)} 个）：", level=LogLevel.INFO)
                for i, q in enumerate(fallback_questions, 1):
                    self.logger.log(f"  {i}. {q}", level=LogLevel.INFO)
                return fallback_questions
            
            total_count = len(all_questions)
            self.logger.log(f"✅ 成功分解为 {total_count} 个子问题", level=LogLevel.INFO)
            for i, q in enumerate(all_questions, 1):
                self.logger.log(f"  {i}. {q}", level=LogLevel.INFO)
            
            # 直接返回全部子问题，数量由 Skill 层决定（task_decompose 生成6-12个）
            return all_questions
        
        except Exception as e:
            self.logger.log(f"⚠️ task_decompose Skill 调用失败: {e}，使用简单分解", level=LogLevel.INFO)
            fallback_questions = self._simple_decompose(task)
            self.logger.log(f"📋 简单分解结果（{len(fallback_questions)} 个）：", level=LogLevel.INFO)
            for i, q in enumerate(fallback_questions, 1):
                self.logger.log(f"  {i}. {q}", level=LogLevel.INFO)
            return fallback_questions

    def _simple_decompose(self, task: str) -> list[str]:
        """简单的问题分解降级方案"""
        return [
            f"什么是{task.split('关于')[1].split('的')[0] if '关于' in task else 'this'}及其核心概念？",
            f"{task.split('关于')[1].split('的')[0] if '关于' in task else 'this'}的历史发展如何？",
            f"{task.split('关于')[1].split('的')[0] if '关于' in task else 'this'}的最新进展是什么？",
            f"{task.split('关于')[1].split('的')[0] if '关于' in task else 'this'}存在哪些主要挑战和机遇？",
        ]

    def _retrieve_for_subquestion(self, sub_question: str) -> str:
        """为单个子问题检索相关材料
        
        支持多种模式（由 self.coarse_rag_mode 控制）：
        - 'web_search': 使用web_search工具
        - 'knowledge': 仅依赖模型内部知识
        - 'disabled': 不进行粗细力度RAG检索
        """
        try:
            # 检查是否禁用粗细力度RAG
            if self.coarse_rag_mode == 'disabled':
                self.logger.log(f"⏸️ 粗细力度RAG已禁用，子问题【{sub_question[:50]}...】将依赖大模型内部知识", level=LogLevel.DEBUG)
                return ""
            
            # 模式：仅使用模型知识
            if self.coarse_rag_mode == 'knowledge':
                self.logger.log(f"📚 使用模型内部知识处理子问题【{sub_question[:50]}...】", level=LogLevel.DEBUG)
                return ""
            
            # 模式：web_search
            if self.coarse_rag_mode != 'web_search':
                self.logger.log(f"⚠️ 未知的粗细力度RAG模式: {self.coarse_rag_mode}", level=LogLevel.INFO)
                return ""
            
            search_query = (sub_question or "").strip()
            if not search_query:
                return ""
            
            self.logger.log(f"🔎 为子问题【{sub_question[:50]}...】调用web_search", level=LogLevel.DEBUG)
            
            try:
                # DuckDuckGoSearchTool 使用 query 参数
                result = self.execute_tool_call(self.search_tool_name, {"query": search_query})
                if result:
                    result_str = str(result)
                    result_len = len(result_str)
                    self.logger.log(f"✅ 检索成功: {self.search_tool_name} ({result_len} 字)", level=LogLevel.INFO)
                    # 打印检索结果摘要（前300字）
                    summary = result_str[:300] + "..." if result_len > 300 else result_str
                    self.logger.log(f"   📄 检索摘要:\n{summary}", level=LogLevel.DEBUG)
                    return result_str
            except Exception as e:
                self.logger.log(f"  ⚠️ {self.search_tool_name} 检索失败: {e}", level=LogLevel.DEBUG)
            
            # 如果web_search不可用，记录警告
            self.logger.log(f"⚠️ web_search工具不可用，该子问题【{sub_question[:50]}...】将依赖大模型内部知识", level=LogLevel.INFO)
            return ""
        
        except Exception as e:
            self.logger.log(f"⚠️ web_search调用异常: {e}", level=LogLevel.DEBUG)
            return ""

    def _build_outline_input(self, sub_questions: list[str], retrieval_results: dict[str, str]) -> str:
        """综合子问题和检索结果，构建大纲生成的输入"""
        self.logger.log("📊 汇总规划输入...", level=LogLevel.INFO)
        
        cleaned_task = self._clean_task_description()
        input_text = f"写作任务：{cleaned_task}\n\n"
        
        input_text += "=== 问题分解 ===\n"
        for i, q in enumerate(sub_questions, 1):
            input_text += f"{i}. {q}\n"
        
        # 统计检索结果
        retrieved_count = sum(1 for r in retrieval_results.values() if r)
        self.logger.log(f"📚 检索统计: {retrieved_count}/{len(sub_questions)} 个子问题有检索结果", level=LogLevel.INFO)
        
        input_text += "\n=== 检索结果 ===\n"
        for q, result in retrieval_results.items():
            if result:
                result_len = len(result)
                input_text += f"\n关于「{q}」的检索结果（{result_len} 字符）：\n"
                self.logger.log(f"   📄 {q[:50]}... ({result_len} 字)", level=LogLevel.DEBUG)
                # 截断过长的检索结果
                trimmed = self._trim_text(result, self.max_outline_context_chars // len(retrieval_results))
                input_text += trimmed
            else:
                input_text += f"\n关于「{q}」: [无检索结果]\n"
                self.logger.log(f"   ⚠️ {q[:50]}... [无检索结果]", level=LogLevel.DEBUG)
        
        total_chars = len(input_text)
        self.logger.log(f"✅ 规划输入准备完成（总长: {total_chars} 字）", level=LogLevel.INFO)
        
        return input_text
    

        # 根据粗细力度RAG模式决定是否进行web搜索
        if self.coarse_rag_mode == 'web_search' and self.enable_coarse_rag_web_search:
            try:
                self.logger.log(f"尝试调用web_search工具", level=LogLevel.DEBUG)
                # DuckDuckGoSearchTool 使用 query 参数
                result = self.execute_tool_call(self.search_tool_name, {"query": rag_query})
                if result and isinstance(result, str) and result.strip():
                    self.logger.log(f"✅ web_search成功", level=LogLevel.INFO)
                    self.state["coarse_rag_tool"] = self.search_tool_name
                    self.state["coarse_rag_context"] = result
                    return result
            except Exception as e:
                self.logger.log(f"⚠️ web_search调用失败: {e}，回退为仅基于任务生成大纲", level=LogLevel.INFO)
        else:
            self.logger.log(f"⏸️ 粗细力度RAG (web_search) 已禁用或模式不匹配，回退为仅基于任务生成大纲", level=LogLevel.INFO)

        self.state["coarse_rag_tool"] = None
        self.state["coarse_rag_context"] = ""
        return ""
    
    def _outline_reflection_loop(self, outline: str, sub_questions: list[str] = None, retrieval_results: dict[str, str] = None) -> str:
        """大纲多轮反思，包含子问题覆盖性校验
        
        Args:
            outline: 大纲文本
            sub_questions: 分解的子问题列表（用于覆盖性校验）
            retrieval_results: 各子问题的检索结果（用于内容覆盖性验证）
        """
        current = outline
        
        for i in range(self.outline_max_iter):
            self.logger.log(f"  大纲反思 {i+1}/{self.outline_max_iter}", level=LogLevel.DEBUG)
            
            try:
                # 构建反思输入（包含子问题列表和检索结果）
                reflection_input = f"大纲:\n{current}"
                
                if sub_questions:
                    reflection_input += "\n\n【子问题列表】需要确保大纲覆盖以下所有问题：\n"
                    for idx, sq in enumerate(sub_questions, 1):
                        reflection_input += f"{idx}. {sq}\n"
                    reflection_input += "\n请检查大纲是否覆盖了所有子问题的核心内容。"
                
                # 加入检索结果（完整保留）
                if retrieval_results:
                    reflection_input += "\n\n【各子问题的检索结果】\n"
                    for idx, (q, result) in enumerate(retrieval_results.items(), 1):
                        if result:
                            reflection_input += f"{idx}. {q}\n{result}\n"
                        else:
                            reflection_input += f"{idx}. {q}\n   [无检索结果]\n"
                    reflection_input += "\n请确保大纲的各章节能够充分利用或覆盖这些检索到的内容。"
                
                # 反思
                report = self.execute_tool_call("outline_reflection", {"input": reflection_input})
                report = str(report)  # 确保返回值是字符串
                data = self._parse_json(report)
                score = data.get("score", 0)
                is_pass = data.get("is_pass", False)
                coverage_of_subquestions = data.get("coverage_of_subquestions", False)
                
                # 写入日志文件
                self._log_reflection_to_file(f"大纲反思第{i+1}轮", score, report)
                
                # 判断是否通过：
                # 1. is_pass 为 true 且 coverage_of_subquestions 为 true，OR
                # 2. score > 80 且 coverage_of_subquestions 为 true
                if coverage_of_subquestions and (is_pass or score > 80):
                    self.logger.log(f"  ✅ 得分 {score}，子问题覆盖 ✓，大纲通过", level=LogLevel.DEBUG)
                    return current
                elif not coverage_of_subquestions:
                    self.logger.log(f"  ⚠️ 得分 {score}，但子问题覆盖不完整，需要修订", level=LogLevel.DEBUG)
                else:
                    self.logger.log(f"  ⚠️ 得分 {score}，未达到通过标准，需要修订", level=LogLevel.DEBUG)
                
                # 修正 - 明确要求输出最终版本，不带修改标记
                revision_prompt = f"""大纲:
{current}

评审报告:
{report}"""
                
                # 在修订时也参考完整的检索结果
                if retrieval_results:
                    revision_prompt += "\n\n【参考：原始检索结果】\n"
                    for q, result in retrieval_results.items():
                        if result:
                            revision_prompt += f"【{q}】\n{result}\n\n"
                
                revision_prompt += """

重要提示：请直接输出修改后的完整大纲，不要使用任何修改标记（如~~删除线~~、**加粗**等），只输出最终的干净文本。在修订时务必参考检索结果确保大纲各章节能够涵盖这些内容。"""
                
                current = str(self.execute_tool_call(
                    "outline_revision",
                    {"input": revision_prompt}
                ))
                
                # 将修改后的内容写入日志
                self._log_revision_to_file(f"大纲修订第{i+1}轮", current)
            
            except Exception as e:
                self.logger.log(f"  ⚠️ 反思出错: {e}", level=LogLevel.INFO)

                return current
        
        return current
    
    # ============== 阶段 2：写作 ==============
    
    def _writing_phase(self, memory_step: ActionStep) -> None:
        """逐段写作 + 多轮证据反思"""
        self.logger.log_rule("✍️ Writing Phase", level=LogLevel.INFO)
        
        sections = self._parse_sections(self._current_outline)
        total_sections = len(sections)
        self.logger.log(f"解析出 {total_sections} 章", level=LogLevel.INFO)
        
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
                self.logger.log(f"📝 [{current_section_num}/{total_sections}] {section['title']} [{section_type}]", level=LogLevel.INFO)

                # 仅在正文章节调用细粒度 RAG，避免额外 token 占用
                fine_rag_context = ""
                available_citations = {}  # 本段落可用的引用
                if section_type == "body":
                    fine_rag_context = self._run_fine_rag_for_section(section)
                    # 从检索结果中提取可用的引用
                    if fine_rag_context:
                        available_citations = self._extract_citations_from_rag(fine_rag_context)

                # 根据类型选择 skill，并按章节语义传最小必要上下文
                # 获取字数目标（如果有）
                word_count_target = section.get('word_count_target', 0)
                
                if section_type == "references":
                    content = self._write_references_section(section)
                    final = content
                elif section_type == "introduction":
                    intro_input = self._build_section_input(section, section_type, fine_rag_context, available_citations)
                    content = self.execute_tool_call(
                        "introduction_write",
                        {"input": intro_input}
                    )
                    final = self._section_reflection_loop(content, section, word_count_target)
                elif section_type == "conclusion":
                    conclusion_input = self._build_section_input(section, section_type, fine_rag_context, available_citations)
                    content = self.execute_tool_call(
                        "conclusion_write",
                        {"input": conclusion_input}
                    )
                    final = self._section_reflection_loop(content, section, word_count_target)
                else:
                    # 使用两步 Skills 管道处理 body 章节（已移除 data_evidence_extraction 冗余步骤）
                    # fine_rag 已提供 5-8 个知识点，无需再提纯
                    # Step 1: 骨架规划 (Skill A)
                    skeleton_input = self._build_skeleton_planning_input(section, fine_rag_context)
                    try:
                        skeleton = self.execute_tool_call(
                            "section_skeleton_planning",
                            {"input": skeleton_input}
                        )
                        self.logger.log(f"✅ 骨架规划完成", level=LogLevel.DEBUG)
                    except Exception as e:
                        self.logger.log(f"⚠️ 骨架规划失败: {e}，降级使用原始 section_write", level=LogLevel.INFO)
                        # 降级：使用原有的 section_write
                        body_input = self._build_section_input(section, section_type, fine_rag_context, available_citations)
                        content = self.execute_tool_call("section_write", {"input": body_input})
                        final = self._section_reflection_loop(content, section, word_count_target)
                        self._generated_sections.append({
                            "title": section['title'],
                            "content": final,
                            "type": section_type
                        })
                        self._update_memory(section['title'], final)
                        continue
                    
                    # Step 2: 文本组装 (Skill B)
                    composition_input = self._build_composition_input(section, fine_rag_context, skeleton, word_count_target)
                    try:
                        content = self.execute_tool_call(
                            "section_composition_styling",
                            {"input": composition_input}
                        )
                        self.logger.log(f"✅ 文本组装完成", level=LogLevel.DEBUG)
                    except Exception as e:
                        self.logger.log(f"⚠️ 文本组装失败: {e}，降级为骨架展开", level=LogLevel.INFO)
                        content = skeleton  # 降级使用骨架作为内容
                    
                    # Step 3: 反思循环检查
                    final = self._section_reflection_loop(content, section, word_count_target)
                
                # 更新内存（参考文献不参与正文记忆）
                if section_type != "references":
                    self._update_memory(section['title'], final)
                    # 添加该段落中的可用引用到全局引用字典
                    self._add_citations(available_citations)
                
                self._generated_sections.append({
                    "title": section['title'],
                    "content": final,
                    "type": section_type
                })
                
                # 👇 关键：立即保存到文件，防止丢失
                self._append_section_to_file(
                    section['title'], 
                    final, 
                    current_section_num, 
                    total_sections,
                    section_type
                )
                section_len = len(str(final))
                self.logger.log(f"✅ [{current_section_num}/{total_sections}] {section['title']} 完成 ({section_len} 字)", level=LogLevel.INFO)
            
            except Exception as e:
                self.logger.log(f"⚠️ [{current_section_num}/{total_sections}] {section['title']} 失败: {e}", level=LogLevel.ERROR)

    def _run_fine_rag_for_section(self, section: Dict[str, str]) -> str:
        """细粒度RAG检索，支持多种模式
        
        模式说明（由 self.fine_rag_mode 控制）：
        - 'web_agent': 调用text_webbrowser_agent进行深度检索（推荐）
        - 'web_search': 使用DuckDuckGo搜索（快速但内容有限）
        - 'disabled': 禁用细粒度RAG
        """
        section_title = section.get('title', '未命名章节')
        section_goal = section.get('goal', '')
        
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
            return self._run_fine_rag_web_search(section_title, search_query)
        
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
            # 构建发送给text_webbrowser_agent的任务
            task_prompt = f"""请为以下章节检索详细信息：

【章节标题】{section_title}
【搜索关键词】{search_query}

【任务要求】：
1. 使用网络搜索查找相关的详细信息和资料
2. 访问搜索结果中最相关的网页，提取其中的具体内容和数据
3. 整理收集到的信息，返回最相关和有用的内容
4. 重点提取能够支撑这个章节的具体事实、数据、理论等

请以清晰、结构化的方式返回收集到的信息。"""

            self.logger.log(f"📍 调用 text_webbrowser_agent 进行深度检索...", level=LogLevel.DEBUG)
            
            # 调用 text_webbrowser_agent（managed agent）
            result = self.execute_tool_call(
                self.text_webbrowser_agent_name,
                {"task": task_prompt}
            )
            
            if isinstance(result, str) and result.strip():
                result_str = str(result)
                result_len = len(result_str)
                self.logger.log(f"✅ text_webbrowser_agent 返回内容 ({result_len} 字)", level=LogLevel.INFO)
                
                # 清理可能的框架包装文本
                cleaned_result = self._clean_agent_output(result_str)
                trimmed = self._trim_text(cleaned_result, self.max_fine_rag_chars)
                return trimmed
            else:
                self.logger.log(f"⚠️ text_webbrowser_agent 无返回内容，回退为web_search", level=LogLevel.INFO)
                return self._run_fine_rag_web_search(section_title, search_query)
        
        except Exception as e:
            self.logger.log(f"⚠️ text_webbrowser_agent 调用异常: {e}，回退为web_search", level=LogLevel.INFO)
            return self._run_fine_rag_web_search(section_title, search_query)
    
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
    
    def _run_fine_rag_web_search(self, section_title: str, search_query: str) -> str:
        """使用DuckDuckGo搜索进行细粒度RAG检索（搜索结果列表）"""
        try:
            self.logger.log(f"🔎 调用DuckDuckGo搜索", level=LogLevel.DEBUG)
            result = self.execute_tool_call(self.search_tool_name, {"query": search_query})
            if isinstance(result, str) and result.strip():
                # 兜底：如果返回了被框架污染的 Task outcome 包装，提取其中有效内容
                raw_result = result.strip()
                markers = [
                    "### 1. Task outcome (short version):",
                    '"### 1. Task outcome (short version)":',
                    "\"final_answer\"",
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
                        result = json_match.group(1).replace('\\n', '\n').strip() if json_match else raw_result
                else:
                    result = raw_result

                result_len = len(result)
                self.logger.log(f"✅ web_search成功 ({result_len} 字)", level=LogLevel.DEBUG)
                trimmed = self._trim_text(result, self.max_fine_rag_chars)
                return trimmed
        except Exception as e:
            self.logger.log(f"  ⚠️ web_search调用失败: {e}", level=LogLevel.DEBUG)
        
        self.logger.log(f"⚠️ web_search不可用，章节【{section_title}】将基于已有上下文生成", level=LogLevel.INFO)
        return ""

    def _build_section_input(self, section: Dict[str, str], section_type: str, fine_rag_context: str = "", available_citations: Dict[str, Dict[str, str]] = None) -> str:
        """按章节语义构造最小必要输入，降低 token 占用并减少模型混淆。"""
        task_text = self._trim_text(self._clean_task_description(), 600)
        outline_text = self._trim_text(self._current_outline, self.max_outline_context_chars)
        prev_text = self._trim_text(self._previous_section_content, self.max_prev_section_chars)
        coarse_text = self._trim_text(self.state.get("coarse_rag_context", ""), 800)
        
        # 获取字数要求
        word_count_target = section.get('word_count_target', 0)
        word_count_hint = ""
        if word_count_target > 0:
            word_count_tolerance = int(word_count_target * 0.2)
            word_count_hint = f"\n\n【字数要求】\n目标字数: {word_count_target}字（允许范围: {word_count_target - word_count_tolerance}-{word_count_target + word_count_tolerance}字）\n请确保输出内容符合字数要求。"
        
        # 格式化可用引用列表
        available_cites_text = ""
        if available_citations:
            available_cites_text = "\n\n【可用引用】\n"
            for key in available_citations.keys():
                available_cites_text += f"- {key}\n"
            available_cites_text += "注：仅在需要时使用上述引用，不要编造新的引用。"

        if section_type == "references":
            bibliography = self._trim_text(self._format_references_section(), self.max_bibliography_chars)
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
            return (
                f"用户prompt:\n{task_text}\n\n"
                f"大纲:\n{outline_text}\n\n"
                f"web_search结果:\n{coarse_text or '（无）'}\n\n"
                f"当前章节: {section.get('title', '')}{available_cites_text}{word_count_hint}"
            )

        # conclusion
        return (
            f"用户prompt:\n{task_text}\n\n"
            f"大纲:\n{outline_text}\n\n"
            f"全文摘要:\n{self._trim_text(self._global_summary, 600) or '（无）'}\n\n"
            f"当前章节: {section.get('title', '')}{available_cites_text}{word_count_hint}"
        )

    def _build_data_extraction_input(self, section: Dict[str, str], rag_context: str) -> str:
        """构造 Skill A (data_evidence_extraction) 的输入"""
        task_text = self._trim_text(self._clean_task_description(), 600)
        
        return (
            f"段落标题：{section.get('title', '')}\n"
            f"段落目标：{section.get('goal', '')}\n\n"
            f"背景资料库：\n{rag_context or '（无）'}\n\n"
            f"用户主题：{task_text}\n\n"
            f"请从背景资料中提取所有与该段落直接相关的数据、事实和论据，\n"
            f"组织成结构化的事实清单，标注每个数据对应的 APA 文内引用（作者, 年份）。"
        )

    def _build_skeleton_planning_input(self, section: Dict[str, str], fact_list: str) -> str:
        """构造 Skill B (section_skeleton_planning) 的输入"""
        outline_text = self._trim_text(self._current_outline, 500)
        prev_summary = self._trim_text(self._previous_section_content[:300], 200)
        
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

    def _build_composition_input(self, section: Dict[str, str], fact_list: str, skeleton: str, word_count_target: int = 0) -> str:
        """构造 Skill C (section_composition_styling) 的输入"""
        word_count_hint = ""
        if word_count_target > 0:
            word_count_tolerance = int(word_count_target * 0.2)
            word_count_hint = (
                f"\n\n【字数要求】\n"
                f"目标字数: {word_count_target}字\n"
                f"允许范围: {word_count_target - word_count_tolerance}-{word_count_target + word_count_tolerance}字"
            )
        
        return (
            f"段落标题：{section.get('title', '')}\n"
            f"风格要求：冷静、笃定、权威、数据驱动（麦肯锡风格）{word_count_hint}\n\n"
            f"逻辑骨架：\n{skeleton}\n\n"
            f"事实清单：\n{fact_list}\n\n"
            f"请严格按照骨架的逻辑顺序，用顶级咨询公司的语言风格，\n"
            f"将数据融合为完整、流畅的段落文本。\n"
            f"确保：\n"
            f"- 遵循骨架的 3-5 句逻辑顺序\n"
            f"- 所有关键数据都有 APA 文内标注（作者, 年份）\n"
            f"- 避免第一人称和模糊表述\n"
            f"- 使用冷静、权威的表达方式"
        )

    def _extract_citations_from_rag(self, rag_context: str) -> Dict[str, Dict[str, str]]:
        """从RAG检索结果中提取可用的引用
        
        返回格式: {
            "Author (Year)": {"authors": "Author", "year": "Year", "title": "..."}
        }
        """
        citations = {}
        if not rag_context:
            return citations
        
        # 匹配 (Author, 2023) / (Smith et al., 2021)
        pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
        matches = re.findall(pattern, str(rag_context))
        
        placeholder_authors = {
            "张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"
        }
        
        for author_raw, year_raw in matches:
            author = author_raw.strip()
            year = year_raw.strip()
            
            # 过滤占位/虚构作者
            author_norm = author.lower().replace(" ", "")
            if author in placeholder_authors or author_norm in placeholder_authors:
                continue
            if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
                continue
            
            key = f"{author} ({year})"
            if key not in citations:
                citations[key] = {
                    "authors": author,
                    "year": year,
                    "title": "Retrieved from search results"
                }
        
        return citations
    
    def _add_citations(self, citations: Dict[str, Dict[str, str]]) -> None:
        """将提取的引用添加到全局引用库（去重）
        
        Args:
            citations: 从RAG结果中提取的引用字典
        """
        for key, info in citations.items():
            if key not in self._citations:
                self._citation_counter += 1
                self._citations[key] = info
    
    def _collect_citations_from_text(self, text: str) -> None:
        """从正文中提取 APA 文内引用并写入内部引用库。"""
        if not text:
            return

        # 匹配 (Author, 2023) / (Smith et al., 2021)
        pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
        matches = re.findall(pattern, str(text))
        if not matches:
            return

        placeholder_authors = {
            "张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"
        }

        for author_raw, year_raw in matches:
            author = author_raw.strip()
            year = year_raw.strip()

            # 过滤占位/虚构作者，避免污染参考文献池
            author_norm = author.lower().replace(" ", "")
            if author in placeholder_authors or author_norm in placeholder_authors:
                continue
            if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
                continue

            key = f"{author} ({year})"
            if key not in self._citations:
                self._citation_counter += 1
                self._citations[key] = {
                    "id": self._citation_counter,
                    "authors": author,
                    "year": year,
                    "title": "Title unavailable",
                }
    
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
    
    def validate_all_citations(self) -> Dict[str, Any]:
        """验证所有收集的引用
        
        实现5步验证流程：
        1. Search - 在学术数据库中检索
        2. Verify - 双源存在性验证
        3. Retrieve - 获取官方BibTeX
        4. Validate - 核对论文摘要与声明匹配
        5. Add - 写入最终参考文献库
        
        Returns:
            验证结果摘要
        """
        if not self.enable_citation_validation:
            self.logger.log(
                "⚠️ 引用验证功能未启用，跳过验证步骤\n"
                "若要启用，请调用 enable_citation_validation_mode(True)",
                level=LogLevel.INFO
            )
            return {'status': 'disabled', 'message': 'Citation validation not enabled'}
        
        if not self._unverified_citations:
            self.logger.log("ℹ️ 没有待验证的引用", level=LogLevel.INFO)
            return {'status': 'empty', 'citations_count': 0}
        
        self.logger.log(
            f"🔍 开始验证 {len(self._unverified_citations)} 个引用...",
            level=LogLevel.INFO
        )
        
        # 按引用唯一性去重
        citations_to_validate = {}
        for author, year, title, claim in self._unverified_citations:
            key = f"{author}_{year}"
            if key not in citations_to_validate:
                citations_to_validate[key] = (author, year, title, claim)
        
        # 执行验证
        validation_results = {}
        for idx, (key, (author, year, title, claim)) in enumerate(citations_to_validate.items(), 1):
            self.logger.log(
                f"[{idx}/{len(citations_to_validate)}] 验证: {author} ({year})",
                level=LogLevel.INFO
            )
            
            # Step 1: Search
            search_results = self._citation_validator.step1_search(author, year, title)
            
            # Step 2: Verify
            verified, verify_info = self._citation_validator.step2_verify(search_results)
            
            if not verified:
                self.logger.log(
                    f"❌ {author} ({year}) 验证失败: 未找到双源",
                    level=LogLevel.INFO
                )
                validation_results[key] = {
                    'status': 'rejected',
                    'reason': 'Failed dual-source verification',
                    'sources_found': verify_info.get('sources_found', [])
                }
                continue
            
            # Step 3: Retrieve
            best_result = verify_info['best_result']
            bibtex = self._citation_validator.step3_retrieve(best_result)
            
            # 创建引用记录
            record = CitationRecord(
                authors=author,
                year=year,
                title=title or best_result.get('title', 'Unknown'),
                source=verify_info['sources_found'][0],
                url=best_result.get('url', ''),
                bibtex=bibtex or '',
                abstract=best_result.get('abstract', ''),
                verification_status='verified',
                claim_text=claim
            )
            
            # Step 4: Validate - 核对摘要与声明
            if record.abstract and claim:
                validation = self._citation_validator.step4_validate(record, claim)
                record.validation_result = validation
                
                # 根据验证结果调整状态
                if not validation.get('is_valid'):
                    record.verification_status = 'rejected'
                    self.logger.log(
                        f"⚠️ {author} ({year}) 摘要与声明不匹配",
                        level=LogLevel.INFO
                    )
            
            # Step 5: Add
            success = self._citation_validator.step5_add(record)
            
            validation_results[key] = {
                'status': record.verification_status,
                'source': record.source,
                'bibtex': record.bibtex,
                'has_abstract': bool(record.abstract),
                'validation': record.validation_result
            }
            
            # 将验证通过的引用更新到本地字典
            if record.verification_status == 'verified':
                self._add_validated_citation(record)
        
        # 保存验证日志
        self._save_citation_validation_log(validation_results)
        
        # 生成验证报告
        report = self._citation_validator.get_validation_report()
        self.logger.log(
            f"✅ 引用验证完成: {report['verified_count']}/{report['total_citations_processed']} 通过验证",
            level=LogLevel.INFO
        )
        
        return {
            'status': 'completed',
            'total': len(citations_to_validate),
            'verified': report['verified_count'],
            'rejected': report['rejected_count'],
            'verification_rate': report['verification_rate'],
            'results': validation_results
        }
    
    def _add_validated_citation(self, record: CitationRecord) -> None:
        """将验证通过的引用添加到本地字典
        
        Args:
            record: 经过验证的引用记录
        """
        key = f"{record.authors} ({record.year})"
        if key not in self._citations:
            self._citation_counter += 1
            self._citations[key] = {
                'authors': record.authors,
                'year': record.year,
                'title': record.title,
                'source': record.source,
                'url': record.url,
                'bibtex': record.bibtex,
                'verified': True,
                'verification_sources': [record.source]
            }
    
    def _save_citation_validation_log(self, validation_results: Dict) -> None:
        """保存引用验证日志到文件
        
        Args:
            validation_results: 验证结果字典
        """
        try:
            import json
            log_data = {
                'timestamp': time.time(),
                'total_citations': len(validation_results),
                'results': validation_results,
                'validator_report': self._citation_validator.get_validation_report()
            }
            
            with open(self._citations_validation_log, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            self.logger.log(
                f"✅ 验证日志已保存到: {self._citations_validation_log}",
                level=LogLevel.INFO
            )
        except Exception as e:
            self.logger.log(f"⚠️ 保存验证日志失败: {e}", level=LogLevel.ERROR)

    def _write_references_section(self, section: Dict[str, str]) -> str:
        """参考文献章节：直接输出收集到的所有引用
        
        不调用 Skill 重新生成，而是使用全文扫描已有的引用
        """
        return self._format_references_section()
    
    def _section_reflection_loop(self, text: str, section: Dict[str, str], word_count_target: int = 0) -> str:
        """段落质量检查：先字数检查 → 再证据反思
        
        优化流程：
        1. 先做快速的本地字数检查（不消耗token）
        2. 字数不达标时，先调整字数再评估证据
        3. 字数达标后，再做证据反思
        """
        current = text
        current_words = len(current)
        section_title = section.get('title', '未命名章节')
        goal = section.get('goal', '')
        
        try:
            # ========== 步骤1: 快速字数检查（本地计算，不调用Tool） ==========
            if word_count_target > 0:
                word_count_tolerance = int(word_count_target * 0.2)  # 允许20%的偏差
                min_words = word_count_target - word_count_tolerance
                max_words = word_count_target + word_count_tolerance
                
                self.logger.log(f"  📏 字数检查: {current_words}字 (目标: {word_count_target}±{word_count_tolerance}字)", level=LogLevel.DEBUG)
                
                # 如果字数不在合理范围，先调整字数
                if current_words < min_words or current_words > max_words:
                    self.logger.log(f"  ⚠️ 字数不达标，先调整字数...", level=LogLevel.INFO)
                    
                    # 构建字数调整提示
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
                    
                    # 调用Tool调整字数
                    current = str(self.execute_tool_call(
                        "section_revision",
                        {"input": word_adjust_prompt}
                    ))
                    
                    # 验证调整后的字数
                    adjusted_words = len(current)
                    self.logger.log(f"  📝 字数调整完成: {current_words}字 → {adjusted_words}字", level=LogLevel.INFO)
                    
                    # 记录到日志文件
                    self._log_revision_to_file(f"字数调整 - {goal[:30]}...", current)
                    
                    # 如果还是不达标，记录警告但继续
                    if adjusted_words < min_words or adjusted_words > max_words:
                        self.logger.log(f"  ⚠️ 字数仍未达标: {adjusted_words}字", level=LogLevel.INFO)
                    else:
                        self.logger.log(f"  ✅ 字数已达标", level=LogLevel.DEBUG)
                    
                    current_words = adjusted_words
                else:
                    self.logger.log(f"  ✅ 字数合格", level=LogLevel.DEBUG)
            
            # ========== 步骤2: 证据反思（字数已合理，专注于质量） ==========
            self.logger.log(f"  🔍 证据反思...", level=LogLevel.DEBUG)
            
            # 清洁化目标：去掉冒号和AI味的格式
            clean_goal = goal.split('|')[0].strip() if '|' in goal else goal
            clean_goal = re.sub(r'[：:].*$', '', clean_goal).strip()
            
            reflection_input = (
                f"章节标题\n\n{section_title}\n\n"
                f"章节目标\n\n{clean_goal}\n\n"
                f"内容\n\n{current}"
            )
            
            report = self.execute_tool_call(
                "evidence_reflection",
                {"input": reflection_input}
            )
            report = str(report)
            data = self._parse_json(report)
            score = data.get("score", 0)
            # is_pass 为 true 或 score > 80 即认为通过（无需再修订）
            evidence_satisfied = data.get("is_pass", False) or (score > 80)
            
            # 写入日志文件
            self._log_reflection_to_file(f"证据反思 - {goal[:30]}...", score, report)
            
            # 检查证据是否充分
            if evidence_satisfied and score >= self.section_score_threshold:
                self.logger.log(f"  ✅ 得分 {score}，证据充分", level=LogLevel.DEBUG)
                return current
            
            # 如果证据不足，进行修订（此时字数已经合理）
            self.logger.log(f"  ⚠️ 得分 {score}，证据不足，进行修订...", level=LogLevel.INFO)
            
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
            
            current = str(self.execute_tool_call(
                "section_revision",
                {"input": evidence_revision_prompt}
            ))
            
            final_words = len(current)
            self.logger.log(f"  ✅ 证据修订完成: {final_words}字", level=LogLevel.DEBUG)
            
            # 将修改后的内容写入日志
            self._log_revision_to_file(f"证据修订 - {goal[:30]}...", current)

        
        except Exception as e:
            self.logger.log(f"  ⚠️ 反思出错: {e}", level=LogLevel.INFO)
        
        return current
    
    # ============== 阶段 3：整合 ==============
    
    def _finalization_phase(self, memory_step: ActionStep) -> None:
        """整合最终输出"""
        self.logger.log_rule("🎁 Finalization", level=LogLevel.INFO)
        
        try:
            # 保存到 state
            self.state["outline"] = self._current_outline
            self.state["sections"] = self._generated_sections
            
            # 统计和总结
            total_sections = len(self._generated_sections)
            total_chars = sum(len(str(s.get('content', ''))) for s in self._generated_sections)
            
            self.logger.log(f"📊 生成统计:", level=LogLevel.INFO)
            self.logger.log(f"   总章节数: {total_sections}", level=LogLevel.INFO)
            self.logger.log(f"   总字数: {total_chars}", level=LogLevel.INFO)
            self.logger.log(f"   输出文件: {self._output_file}", level=LogLevel.INFO)
            
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
    
    def _parse_sections(self, outline: str) -> List[Dict[str, str]]:
        """解析大纲为章节，统一用 | 分隔符提取所有元数据（标题、字数、目标、数据等）"""
        self.logger.log("📑 解析大纲章节...", level=LogLevel.DEBUG)
        self.logger.log(f"大纲内容（前500字）:\n{outline[:500]}...", level=LogLevel.DEBUG)
        
        sections = []
        lines = outline.split("\n")
        
        # 统一解析格式：数字编号或 Markdown 格式，都用 | 分隔元数据
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 支持数字编号格式: "1. ", "2.1 ", "3.2.1 " 等
            match_numbered = re.match(r'^([\d.]+)\s+(.+)$', line)
            if match_numbered:
                raw_content = match_numbered.group(2).strip()
                number = match_numbered.group(1)
                self._parse_and_add_section(raw_content, sections, number=number)
                continue
            
            # 支持 Markdown 格式: "# ", "## ", "### " 等
            match_markdown = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match_markdown:
                raw_content = match_markdown.group(2).strip()
                level = len(match_markdown.group(1))
                self._parse_and_add_section(raw_content, sections, level=level)
                continue
        
        # 如果解析失败，尝试容错方案：提取所有非空行作为章节标题
        if not sections:
            self.logger.log("⚠️ 标准格式解析失败，尝试容错方案...", level=LogLevel.INFO)
            for line in lines:
                line = line.strip()
                # 过滤掉明显的说明文字和目标/字数描述行
                if (line and 
                    not line.startswith("- ") and 
                    not line.startswith("目标") and 
                    not "字" in line and 
                    not any(keyword in line for keyword in ["任务", "包含", "要求", "说明", "输出格式"])):
                    
                    # 清理字符
                    title = re.sub(r'[\(\[\-].*', '', line).strip()
                    
                    if title and len(title) > 2:
                        sections.append({
                            "title": title,
                            "goal": title,
                            "word_count_target": 0,
                            "fallback": True
                        })
        
        self.logger.log(f"✅ 解析完成，共 {len(sections)} 个章节", level=LogLevel.INFO)
        if sections:
            self.logger.log("章节列表:", level=LogLevel.INFO)
            for i, sec in enumerate(sections[:10], 1):  # 只显示前10个
                self.logger.log(f"  {i}. {sec['title']}", level=LogLevel.INFO)
            if len(sections) > 10:
                self.logger.log(f"  ... 还有 {len(sections) - 10} 个章节", level=LogLevel.INFO)
        else:
            self.logger.log("⚠️ 未能解析出任何章节！", level=LogLevel.ERROR)
            self.logger.log("问题诊断:", level=LogLevel.ERROR)
            self.logger.log(f"  - 大纲总行数: {len(lines)}", level=LogLevel.ERROR)
            self.logger.log(f"  - 非空行数: {len([l for l in lines if l.strip()])}", level=LogLevel.ERROR)
            self.logger.log(f"  - 完整大纲内容:\n{outline}", level=LogLevel.ERROR)
        
        return sections
    
    def _parse_and_add_section(self, raw_content: str, sections: List, number: str = None, level: int = None):
        """解析单个章节内容，提取标题和所有元数据
        
        格式: 标题 | 字数: 800 | 目标: xxx | 数据: xxx | ...
        或仅: 标题
        """
        metadata = {}
        word_count_target = 0
        
        # 用 | 分隔标题和元数据
        if ' | ' in raw_content:
            parts = [p.strip() for p in raw_content.split(' | ')]
            title = parts[0]
            
            # 解析所有元数据
            for part in parts[1:]:
                if part.startswith('字数'):
                    # 格式: 字数: 800 或 字数:800
                    match = re.search(r'\d+', part)
                    if match:
                        word_count_target = int(match.group())
                elif part.startswith('目标'):
                    metadata['goal'] = re.sub(r'^目标[：:]\s*', '', part)
                elif part.startswith('数据'):
                    metadata['data'] = re.sub(r'^数据[：:]\s*', '', part)
                elif part.startswith('问题'):
                    metadata['question'] = re.sub(r'^问题[：:]\s*', '', part)
                else:
                    # 其他未知的元数据也保存
                    if ':' in part:
                        key, val = part.split(':', 1)
                        metadata[key.strip().lower()] = val.strip()
        else:
            # 没有元数据，仅有标题
            title = raw_content
        
        goal = metadata.get('goal', title)
        
        section_dict = {
            "title": title,
            "goal": goal,
            "word_count_target": word_count_target,
            "metadata": metadata
        }
        
        # 添加可选的位置信息
        if number is not None:
            section_dict["number"] = number
        if level is not None:
            section_dict["level"] = level
        
        sections.append(section_dict)
    
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
        
        # 按作者-年份排序（APA 风格）
        sorted_refs = sorted(
            self._citations.items(),
            key=lambda x: (str(x[1].get('authors', '')).lower(), str(x[1].get('year', '')))
        )
        
        lines = []
        for _, info in sorted_refs:
            authors = info.get('authors', '未知作者')
            year = info.get('year', 'n.d.')
            source_title = info.get('title', 'Title unavailable')
            
            # APA 近似兜底格式
            lines.append(f"{authors}. ({year}). {source_title}.")
        
        return "\n".join(lines)
    
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
