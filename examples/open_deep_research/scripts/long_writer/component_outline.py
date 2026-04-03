from __future__ import annotations
import sys
import json
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..long_writer_agent_v3 import LongWriterAgent

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

    def run(self, agent: "LongWriterAgent", payload: Dict[str, Any]) -> Dict[str, Any]:
        outline_v1 = str(agent.execute_tool_call(
            "outline_generation", 
            {"input": json.dumps(payload, ensure_ascii=False)}
        ))

        return {
            "outline_input": payload,
            "outline_v1": outline_v1
        }

if __name__ == "__main__":
    sys.exit(run_component_cli(OutlineGenerationComponent()))