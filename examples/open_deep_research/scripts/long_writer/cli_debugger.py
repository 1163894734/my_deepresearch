import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import common_utils

# ==========================================
# 🚀 路径魔法：解决直接运行时的报错
# ==========================================
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent

if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 顶层仅导入基础纯契约类，绝不导入复杂的 Agent，打破循环依赖！
try:
    from .base_component import JsonWorkflowComponent
except ImportError:
    from base_component import JsonWorkflowComponent
# ==========================================


def _build_default_input_payload(component: JsonWorkflowComponent) -> Dict[str, Any]:
    """为组件自动构造默认输入载荷"""
    schema = component.input_format or {}
    if not isinstance(schema, dict): return {}
    defaults: Dict[str, Any] = {}
    for key, hint in schema.items():
        text = str(hint or "").lower()
        if "str" in text: defaults[key] = "调试任务：请分析大模型前沿进展" if str(key) == "task" else ""
        elif "list" in text: defaults[key] = []
        elif "dict" in text: defaults[key] = {}
        elif "bool" in text: defaults[key] = False
        else: defaults[key] = ""
    return defaults


def _mock_agent_for_cli():
    """构建一个用于本地调试的轻量级 Agent 实例（接入真实大模型）"""
    
    # 延迟导入 Agent 和 Tool Loader，打破循环引用！
    try:
        from ..long_writer_agent_v3 import LongWriterAgent
        from ..skill_loader import load_skills_from_directory
    except ImportError:
        from long_writer_agent_v3 import LongWriterAgent
        from skill_loader import load_skills_from_directory

    from smolagents import OpenAIModel

    # 同步使用 run.py 中的真实模型配置
    real_model = common_utils.ModelProvider.get_model()
    
    # 动态获取项目根目录并加载 skills 文件夹下的所有工具
    root_dir = Path(__file__).resolve().parent.parent.parent
    skills_dir = root_dir / "skills"
    
    print(f"\n[CLI Debugger] 正在为测试 Agent 加载技能库 (接入真实大模型: {real_model.model_id})...", file=sys.stderr)
    tools = load_skills_from_directory(str(skills_dir), model=real_model)
    
    # 将加载到的工具塞入测试 Agent
    agent = LongWriterAgent(model=real_model, tools=tools)
    
    # 将输出路径重定向到专门的 debug 目录，避免污染正式报告
    debug_dir = f"outputs/debug_{time.strftime('%Y%m%d_%H%M%S')}"
    agent.workspace.output_dir = debug_dir
    agent.workspace.log_file = f"{debug_dir}/cli_debug.log"
    os.makedirs(debug_dir, exist_ok=True)
    
    return agent


def run_component_cli(component: JsonWorkflowComponent, argv: Optional[List[str]] = None) -> int:
    """组件 CLI 调试统一入口"""
    parser = argparse.ArgumentParser(description=f"组件 CLI 调试入口（component={component.name}）")
    parser.add_argument("--input-json", help="输入 JSON 字符串")
    parser.add_argument("--input-file", help="输入 JSON 文件路径")
    parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON")
    
    args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    
    # 1. 解析输入
    raw_payload = None
    if args.input_file:
        raw_payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    elif args.input_json:
        raw_payload = json.loads(args.input_json)
        
    if raw_payload is None:
        input_payload = _build_default_input_payload(component)
        print("[debug] 未提供输入参数，已自动使用默认调试入参。", file=sys.stderr)
    else:
        input_payload = raw_payload.get("input", raw_payload)

    # 2. 组装运行环境
    agent = _mock_agent_for_cli()
    
    # 3. 执行组件逻辑
    try:
        print(f"\n🚀 开始调试组件: {component.name}", file=sys.stderr)
        output = component.run(agent, input_payload)
        
        print("\n✅ 调试完成，输出结果:", file=sys.stderr)
        if isinstance(output, (dict, list)):
            print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))
        else:
            print(output)
        return 0
    except Exception as e:
        print(f"\n❌ 组件运行失败: {e}", file=sys.stderr)
        return 1