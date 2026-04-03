from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..long_writer_agent_v3 import LongWriterAgent

from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli
from .section_writing_service import SectionWritingService

class ConclusionWritingComponent(JsonWorkflowComponent):
    """结论撰写组件：输入章节与上下文，输出通过反思后的结论终稿。"""

    name = "conclusion_writing"
    description = "总结全文贡献与展望，并进行内联引用校验。"
    input_format = {
        "section": "Dict[str, Any]",
        "available_citations": "Dict[str, Dict[str, Any]]",
        "full_text": "str",
        "task": "str",
    }
    output_format = {
        "content": "str",
        "section_ref": "str",
    }

    def run(self, agent: "LongWriterAgent", payload: Dict[str, Any]) -> Dict[str, Any]:
        final = SectionWritingService.write_conclusion(agent, payload)
        section = payload.get("section", {}) if isinstance(payload, dict) else {}
        section_ref = section.get("title", "结论") if isinstance(section, dict) else "结论"
        
        raw_citations = payload.get("available_citations", {}) if isinstance(payload, dict) else {}
        if raw_citations and not isinstance(raw_citations, dict):
            raise ValueError("available_citations 必须是 Dict[\"title\", citation_info]，不再接受 list")
        available_citations = raw_citations if isinstance(raw_citations, dict) else {}

        if hasattr(agent, "_run_inline_citation_validation_for_section"):
            final = agent._run_inline_citation_validation_for_section(
                section_type="conclusion",
                section_ref=str(section_ref),
                section_content=str(final),
                available_citations=available_citations,
            )
        return {"content": str(final), "section_ref": str(section_ref)}

if __name__ == "__main__":
    sys.exit(run_component_cli(ConclusionWritingComponent()))