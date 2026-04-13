import os
import threading
from datetime import datetime
from pathlib import Path

from scripts.long_writer.registry import build_workflow_components
from utils.common_utils import execute_tool_call
# 🔥 把被我不小心弄丢的 import 加回来了！
from scripts.skill_loader import load_skills_from_directory


class DummyLogger:
    """
    带文件持久化和线程锁的终极日志器。
    """
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path
        self._lock = threading.Lock()
        
        if self.log_file_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write(f"=== Report Generation Log Started at {datetime.now()} ===\n\n")

    def log(self, msg, *args, **kwargs):
        level = kwargs.get('level')
        if not level and args:
            level = args[0]
        prefix = f"[{level}] " if level else "[INFO] "
        log_line = f"{prefix}{msg}"
        
        # 1. 打印到终端
        print(log_line)
        
        # 2. 线程安全地写到文件
        if self.log_file_path:
            try:
                with self._lock:
                    with open(self.log_file_path, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")
            except Exception:
                pass
        
    def info(self, msg, *args, **kwargs): self.log(msg, "INFO", *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.log(msg, "WARNING", *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.log(msg, "ERROR", *args, **kwargs)
    def debug(self, msg, *args, **kwargs): pass
    
    def __getattr__(self, name):
        # 拦截所有不存在的 logger 方法
        return lambda *args, **kwargs: None


class PipelineContext:
    """
    流水线全局上下文 (Context) - 完美伪装者与黑洞防御模式
    """
    def __init__(self, model, workspace):
        self.model = model
        self.workspace = workspace
        
        # 🔥 获取输出目录并初始化带锁的 Logger
        log_path = os.path.join(workspace.output_dir, "generation_log.txt")
        self.logger = DummyLogger(log_file_path=log_path)
        
        self.components = build_workflow_components()
        
        # 1. 核心工具与状态
        self.state = {"search_engine": "arxiv"}
        self.managed_agents = {}
        self.tools = self._load_all_skills(model)
        
        # 2. 完美克隆老 Agent 的所有状态变量
        self._citations = {} 
        self._citation_counter = 1
        self._queued_citation_keys = set()           
        self._unverified_citations = []              
        
        self._current_outline = ""
        self._generated_sections = []
        self._abstract_section = None
        self._previous_section_content = ""
        self._prev_body_or_intro_content = ""
        self._global_summary = ""
        self._full_text_body = ""
        
        # 3. 各种玄学配置参数占位
        self.enable_citation_validation = True
        self.citation_validation_timeout = 5
        self.outline_max_iter = 1
        self.section_max_iter = 1
        self.outline_score_threshold = 90
        self.section_score_threshold = 88
        self.max_summary_length = 300
        self._language = "Chinese"
        self._task = ""
        
    def _load_all_skills(self, model) -> dict:
        current_dir = Path(__file__).resolve().parent
        skills_dir = current_dir.parent.parent / "skills"
        loaded_tools_dict = {}
        if skills_dir.exists():
            self.logger.info(f"🔧 正在加载技能集从: {skills_dir}")
            for t in load_skills_from_directory(str(skills_dir), model=model):
                loaded_tools_dict[t.name] = t
            self.logger.info(f"✅ 成功加载 {len(loaded_tools_dict)} 个技能工具。")
        return loaded_tools_dict
        
    # ==========================================
    # 魔法防御层 (Magic Methods)
    # ==========================================
    def __getattr__(self, name):
        if name.startswith('_') and ("write" in name or "log" in name or "save" in name):
            return lambda *args, **kwargs: None
            
        self.logger.warning(f"🛡️ [Context黑洞防御] 拦截到未知属性访问: {name}，已安全返回 None。")
        return None

    # ==========================================
    # 保留关键的 IO 委托
    # ==========================================
    def _safe_write_file(self, *args, **kwargs): self.workspace.safe_write(*args, **kwargs)
    def _append_section_to_file(self, *args, **kwargs): self.workspace.append_section(*args, **kwargs)
    def _get_heading_prefix(self, *args, **kwargs): return self.workspace.get_heading_prefix(*args, **kwargs)
    def _save_outline_to_file(self, *args, **kwargs): self.workspace.save_outline(*args, **kwargs)
    def _log_reference_event(self, *args, **kwargs): self.workspace.log_reference_event(*args, **kwargs)
    def _append_citation_validation_text_log(self, *args, **kwargs): self.workspace.append_citation_validation_text_log(*args, **kwargs)

    def execute_tool_call(self, tool_name: str, arguments: dict):
        import json
        import time
        from datetime import datetime
        # 注意：这里调用的是底层的原生方法，不再用 self.tools 防止递归
        from utils.common_utils import execute_tool_call as base_execute
        
        # ==========================================
        # 1. 组装详细的输入 Trace
        # ==========================================
        trace_msg = f"\n{'='*80}\n"
        trace_msg += f"🛠️ [TOOL CALL START] {tool_name}\n"
        trace_msg += f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        trace_msg += f"{'-'*80}\n"
        
        # 🔥 神仙操作：反向提取 PromptSkillTool 里的 System Prompt
        tool = self.tools.get(tool_name)
        if tool and hasattr(tool, '_skill_md_path') and tool._skill_md_path:
            try:
                from scripts.skill_loader import _read_skill_prompt
                sys_prompt = _read_skill_prompt(tool._skill_md_path)
                trace_msg += f"🧠 [System Prompt (SKILL.md)]:\n{sys_prompt.strip()}\n\n"
            except Exception:
                pass
                
        # 记录用户传入的参数
        input_str = json.dumps(arguments, ensure_ascii=False, indent=2)
        trace_msg += f"📥 [User Input / Arguments]:\n{input_str}\n"
        trace_msg += f"{'-'*80}\n"
        
        # 写入 Trace 文件，终端只打印简要进度
        self.workspace.safe_write("detailed_trace.log", trace_msg, mode="a")
        self.logger.info(f"▶️ 开始执行工具: {tool_name}")
        
        # ==========================================
        # 2. 执行底层调用并记录输出
        # ==========================================
        start_t = time.time()
        try:
            result = base_execute(tool_name, arguments, self.tools, self.logger)
            cost_t = time.time() - start_t
            
            # 记录成功输出
            out_msg = f"📤 [Tool Output] (耗时 {cost_t:.2f}s):\n{str(result)}\n"
            out_msg += f"{'='*80}\n\n"
            self.workspace.safe_write("detailed_trace.log", out_msg, mode="a")
            self.logger.info(f"✅ 工具执行成功: {tool_name} ({cost_t:.2f}s)")
            
            return result
        except Exception as e:
            # 记录失败输出
            err_msg = f"❌ [Tool Error]: {str(e)}\n{'='*80}\n\n"
            self.workspace.safe_write("detailed_trace.log", err_msg, mode="a")
            self.logger.error(f"❌ 工具执行失败: {tool_name} - {e}")
            raise