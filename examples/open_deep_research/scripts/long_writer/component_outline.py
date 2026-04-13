from __future__ import annotations
import sys
import json
from typing import TYPE_CHECKING, Any, Dict
from utils.common_utils import execute_tool_call

if TYPE_CHECKING:
    # 引入我们新的上下文对象进行类型提示
    from scripts.multi_agent.agent_context import PipelineContext

from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli

class OutlineGenerationComponent(JsonWorkflowComponent):
    """大纲生成与反思修订组件。"""

    name = "outline_generation"
    description = "输入检索证据与任务要求，调用大模型生成最终大纲。"
    input_format = {
        "task": "str",
        "available_citations": "Dict[str, Dict[str, Any]]"
    }
    output_format = {
        "outline_input": "Dict[str, Any]",
        "outline_v1": "str",
    }

    def run(self, context: "PipelineContext", payload: Dict[str, Any]) -> Dict[str, Any]:
        available_tools = {**getattr(context, "tools", {}), **getattr(context, "managed_agents", {})}
        outline_v1 = str(execute_tool_call(
            "outline_generation", 
            {"input": json.dumps(payload, ensure_ascii=False)},
            available_tools=available_tools,
            logger=getattr(context, "logger", None),
        ))

        return {
            "outline_input": payload,
            "outline_v1": outline_v1
        }

if __name__ == "__main__":
    sys.exit(run_component_cli(OutlineGenerationComponent()))