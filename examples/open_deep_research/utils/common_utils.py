# 文件路径: open_deep_research/scripts/utils/common_utils.py

import json
import re
from typing import Any, Dict, List, Union
import time
import logging
import sys
from smolagents.monitoring import LogLevel

def safe_json_parse(text: str, fallback_type: type = dict) -> Union[Dict, List, Any]:
    """
    终极 JSON 解析工具：处理 Markdown、前后废话以及复杂的嵌套结构。
    """
    if not text or not isinstance(text, str):
        return fallback_type()
        
    text = text.strip()
    
    # 尝试 1：直接解析（最快乐的路径）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 尝试 2：剥离 Markdown 代码块干扰
    clean_text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass
        
    # 尝试 3：首尾括号定位法（完美解决嵌套 JSON 与后续废话的问题）
    # 寻找第一个 { 或 [ 
    start_obj = clean_text.find('{')
    start_arr = clean_text.find('[')
    
    # 确定究竟是解析字典还是列表
    is_obj = start_obj != -1 and (start_arr == -1 or start_obj < start_arr)
    start_idx = start_obj if is_obj else start_arr
    
    if start_idx != -1:
        end_char = '}' if is_obj else ']'
        # 从后往前找最后一个对应的闭合括号
        end_idx = clean_text.rfind(end_char) 
        
        if end_idx != -1 and end_idx > start_idx:
            candidate = clean_text[start_idx:end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
                
    # 全都失败，返回保底类型
    return fallback_type()
import os
from smolagents import OpenAIModel

class ModelProvider:
    _instances = {}

    @classmethod
    def get_model(cls, model_type="main"):
        """
        获取模型单例。支持 'main' (大模型) 和 'small' (小模型)
        """
        if model_type not in cls._instances:
            if model_type == "main":
                cls._instances[model_type] = OpenAIModel(
                    model_id="Qwen3-235B-A22B-Instruct-2507",
                    api_base="https://llmapi.paratera.com/v1",
                    api_key=os.environ.get("DYM_API_KEY", ""),
                    max_tokens=8192
                )
            elif model_type == "small":
                # 对应 run_agent3.py 中的本地小模型逻辑
                cls._instances[model_type] = OpenAIModel(
                    model_id="local-model",
                    api_base="http://localhost:1234/v1",
                    api_key="not-needed",
                )
        return cls._instances[model_type]

    @classmethod
    def clear_cache(cls):
        cls._instances = {}

def execute_tool_call(tool_name: str, arguments: Dict[str, Any], available_tools: dict, logger=None) -> Any:
    """
    统一的纯函数工具调用器，不再依赖具体的 Agent 实例。
    
    :param tool_name: 工具名称
    :param arguments: 传递给工具的参数字典
    :param available_tools: 可用工具的字典映射，例如 {"web_search": WebSearchTool()}
    :param logger: 可选的日志记录器
    """
    if tool_name not in available_tools:
        error_msg = f"未找到名为 '{tool_name}' 的工具。"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    tool = available_tools[tool_name]
    start_time = time.perf_counter()
    
    try:
        if logger:
            logger.info(f"🛠️ [Tool执行] 开始调用: {tool_name}")
            
        # smolagents 的 Tool 类通常通过 __call__ 或 forward 执行
        # 如果你的自定义 tool 是普通的 Python 函，直接 tool(**arguments) 即可
        if hasattr(tool, "forward"):
            result = tool.forward(**arguments)
        else:
            result = tool(**arguments)
            
        cost_ms = int((time.perf_counter() - start_time) * 1000)
        if logger:
            logger.info(f"✅ [Tool成功] {tool_name} 耗时 {cost_ms}ms")
            
        return result
        
    except Exception as e:
        cost_ms = int((time.perf_counter() - start_time) * 1000)
        if logger:
            logger.error(f"❌ [Tool失败] {tool_name} 耗时 {cost_ms}ms，错误: {e}")
        raise

class PipelineLogger:
    """
    专为多智能体并发框架设计的标准 Logger。
    完美兼容 smolagents 底层的 agent.logger.log() 调用。
    """
    def __init__(self, agent_name: str, log_file: str = None):
        self.logger = logging.getLogger(agent_name)
        
        # 避免重复添加 Handler 导致日志打印多次
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # 定义高可读性的日志格式（加上了线程/Agent名字区分并发上下文）
            formatter = logging.Formatter(
                '%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 1. 输出到控制台
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 2. (可选) 输出到文件，用于赛后复盘
            if log_file:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    # ==========================================
    # 兼容 smolagents 专属接口
    # ==========================================
    def log(self, msg: str, level=None):
        """兼容底层 smolagents.monitoring.LogLevel 的调用"""
        if level == LogLevel.ERROR:
            self.logger.error(msg)
        elif level == LogLevel.DEBUG:
            self.logger.debug(msg)
        elif level == LogLevel.WARNING:
            self.logger.warning(msg)
        else:
            self.logger.info(msg)

    # ==========================================
    # 兼容标准 Python Logging 接口
    # ==========================================
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)