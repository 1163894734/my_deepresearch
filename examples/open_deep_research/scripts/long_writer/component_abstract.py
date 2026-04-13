from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    # 引入我们新的上下文对象进行类型提示
    from scripts.multi_agent.agent_context import PipelineContext

from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli
from .section_writing_service import SectionWritingService

class AbstractWritingComponent(JsonWorkflowComponent):
    """摘要撰写组件：输入摘要章节配置与结论文本，输出摘要终稿。"""

    name = "abstract_writing"
    description = "根据摘要要求、结论和全文内容生成摘要。"
    input_format = {
        "section": "Dict[str, Any]",
        "conclusion_text": "str",
        "full_text": "str",
        "task": "str",
    }
    output_format = {
        "content": "str",
    }

    def run(self, context: "PipelineContext", payload: Dict[str, Any]) -> Dict[str, Any]:
        final = SectionWritingService.write_abstract(context, payload)
        return {"content": str(final)}

if __name__ == "__main__":
    sys.exit(run_component_cli(AbstractWritingComponent()))