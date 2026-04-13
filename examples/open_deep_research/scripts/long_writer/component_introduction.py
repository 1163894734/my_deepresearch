from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    # 引入我们新的上下文对象进行类型提示
    from scripts.multi_agent.agent_context import PipelineContext

from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli
from .section_writing_service import SectionWritingService

class IntroductionWritingComponent(JsonWorkflowComponent):
    """引言撰写组件：输入章节与上下文，输出通过反思后的引言终稿。"""

    name = "introduction_writing"
    description = "根据章节配置、引用文献、任务和前文内容撰写引言。"
    input_format = {
        "section": "Dict[str, Any]",
        "available_citations": "Dict[str, Dict[str, Any]]",
        "task": "str",
        "prev_section_content": "str",
    }
    output_format = {
        "content": "str",
        "section_ref": "str",
    }

    def run(self, context: "PipelineContext", payload: Dict[str, Any]) -> Dict[str, Any]:
        final = SectionWritingService.write_intro(context, payload)
        section = payload.get("section", {}) if isinstance(payload, dict) else {}
        section_ref = section.get("title", "引言") if isinstance(section, dict) else "引言"
        return {"content": str(final), "section_ref": str(section_ref)}

if __name__ == "__main__":
    sys.exit(run_component_cli(IntroductionWritingComponent()))