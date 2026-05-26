# 文件路径: open_deep_research/scripts/utils/common_utils.py

import json
import re
import ast
import time
from typing import Union, Dict, List, Any, Optional

import re
import json
import ast
from typing import Union, Dict, List, Any
from dotenv import load_dotenv
load_dotenv()
def safe_json_parse(text: str, fallback_type: type = dict) -> Union[Dict, List, Any]:
    """
    终极 JSON 解析工具：处理大模型输出的各种奇葩格式。
    
    能处理的场景：
    - 前后有废话文字
    - Markdown 代码块包裹
    - 单引号/双引号混用
    - 嵌套 JSON 结构
    - Python 字面量格式（True/False/None）
    - 注释干扰
    - 不完整的 JSON（尝试修复）
    
    Args:
        text: 包含 JSON 的原始字符串
        fallback_type: 解析失败时返回的空对象类型（dict 或 list）
    
    Returns:
        解析后的 Python 对象（dict 或 list）
    """
    if not text or not isinstance(text, str):
        return fallback_type() if callable(fallback_type) else fallback_type
    
    original_text = text
    text = text.strip()
    
    # ========== 预处理：清理 LLM 输出的常见干扰 ==========
    
    # 1. 移除 Markdown 代码块标记
    text = re.sub(r'```(?:json|python|py|javascript|js)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'^```\s*', '', text)
    
    # 2. 移除行内注释（// 和 # 开头的行）
    lines = []
    for line in text.split('\n'):
        # 移除行尾的 // 注释（但要小心 URL 中的 //）
        if '//' in line and '://' not in line:
            line = line.split('//')[0]
        # 移除 # 注释（除非在引号内，这里做简化处理）
        if '#' in line and '"' not in line and "'" not in line:
            line = line.split('#')[0]
        lines.append(line)
    text = '\n'.join(lines)
    
    # 3. 标准化布尔值和空值（Python 格式 → JSON 格式）
    # 但要小心不要替换字符串内的内容，这里先用简单规则
    text = re.sub(r'\bTrue\b(?!["\'])', 'true', text)
    text = re.sub(r'\bFalse\b(?!["\'])', 'false', text)
    text = re.sub(r'\bNone\b(?!["\'])', 'null', text)
    
    # ========== 确定解析模式：根据第一个有效字符 ==========
    
    # 找到第一个 { 或 [ 的位置
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    # 情况1：先遇到 [，优先解析为列表
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        return _parse_as_list(text, fallback_type)
    
    # 情况2：先遇到 {，优先解析为字典
    elif first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        return _parse_as_dict(text, fallback_type)
    
    # 情况3：都没有找到，返回保底类型
    return fallback_type() if callable(fallback_type) else fallback_type


def _parse_as_list(text: str, fallback_type: type) -> Union[List, Any]:
    """优先解析为列表"""
    # 策略 1：直接 JSON 解析
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # 策略 2：ast.literal_eval
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError, MemoryError):
        pass
    
    # 策略 3：提取列表内容（从第一个 [ 到匹配的 ]）
    extracted = _extract_bracketed_content(text, '[', ']')
    if extracted is not None:
        try:
            result = json.loads(extracted)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(extracted)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 策略 4：尝试补全不完整的列表
    extracted = _repair_incomplete_brackets(text, '[', ']')
    if extracted is not None:
        try:
            result = json.loads(extracted)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(extracted)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 失败返回保底列表
    return fallback_type() if callable(fallback_type) else fallback_type


def _parse_as_dict(text: str, fallback_type: type) -> Union[Dict, Any]:
    """优先解析为字典"""
    # 策略 1：直接 JSON 解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # 策略 2：ast.literal_eval
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError, MemoryError):
        pass
    
    # 策略 3：提取字典内容（从第一个 { 到匹配的 }）
    extracted = _extract_bracketed_content(text, '{', '}')
    if extracted is not None:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(extracted)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 策略 4：尝试补全不完整的字典
    extracted = _repair_incomplete_brackets(text, '{', '}')
    if extracted is not None:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(extracted)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 失败返回保底字典
    return fallback_type() if callable(fallback_type) else fallback_type


def _extract_bracketed_content(text: str, open_char: str, close_char: str) -> Union[str, None]:
    """提取从第一个 open_char 到匹配的 close_char 之间的内容"""
    start_idx = text.find(open_char)
    if start_idx == -1:
        return None
    
    stack = []
    for i, ch in enumerate(text[start_idx:], start_idx):
        if ch == open_char:
            stack.append(ch)
        elif ch == close_char:
            stack.pop()
            if not stack:
                return text[start_idx:i+1]
    return None


def _repair_incomplete_brackets(text: str, open_char: str, close_char: str) -> Union[str, None]:
    """尝试补全不完整的括号结构"""
    start_idx = text.find(open_char)
    if start_idx == -1:
        return None
    
    # 计算缺少的闭合括号数量
    open_count = text[start_idx:].count(open_char)
    close_count = text[start_idx:].count(close_char)
    missing_count = open_count - close_count
    
    if missing_count > 0 and missing_count < 100:  # 限制补全数量
        return text + close_char * missing_count
    
    return None


def _extract_and_repair_json(text: str) -> Union[Dict, List, None]:
    """（保留原有逻辑的占位函数）智能提取 + 递归修复"""
    # 这里可以保留原有的复杂修复逻辑
    # 为了完整性，简单调用正则提取
    return _regex_extract_json(text)


def _scan_and_complete_braces(text: str) -> Union[Dict, List, None]:
    """（保留原有逻辑的占位函数）逐行扫描 + 括号补全"""
    # 这里可以保留原有的复杂修复逻辑
    return None


def _regex_extract_json(text: str) -> Union[Dict, List, None]:
    """使用正则暴力提取最外层的 JSON 结构"""
    # 尝试提取对象
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(match.group(1))
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 尝试提取数组
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(match.group(1))
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass
    
    return None
import os
from smolagents import OpenAIModel

class ModelProvider:
    _instances = {}

    @classmethod
    def get_model(cls, model_type="main", temperature=None):
        """
        获取模型单例。支持 'main' (大模型) 和 'small' (小模型)
        """
        if model_type not in cls._instances:
            if model_type == "main":
                cls._instances[model_type] = OpenAIModel(
                    model_id=os.environ.get("MODEL_NAME", ""),
                    api_base=os.environ.get("MODEL_BASE_URL", ""),
                    api_key=os.environ.get("MODEL_API_KEY", ""),
                    max_tokens=8192,
                    temperature=temperature if temperature is not None else 0.5
                )
            elif model_type == "small":
                # 对应 run_agent3.py 中的本地小模型逻辑
                cls._instances[model_type] = OpenAIModel(
                    model_id=os.environ.get("MODEL_NAME", ""),
                    api_base=os.environ.get("MODEL_BASE_URL", ""),
                    api_key=os.environ.get("MODEL_API_KEY", ""),
                    temperature=temperature if temperature is not None else 0.5
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