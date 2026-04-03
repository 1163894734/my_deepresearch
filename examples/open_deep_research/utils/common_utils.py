# 文件路径: open_deep_research/scripts/utils/common_utils.py

import json
import re
from typing import Any, Dict, List, Union

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