from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Dict
from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli
from .section_writing_service import SectionWritingService

if TYPE_CHECKING:
    # 引入我们新的上下文对象进行类型提示
    from scripts.multi_agent.agent_context import PipelineContext

class BodyWritingComponent(JsonWorkflowComponent):
    """普通段落撰写组件：实现从骨架到终稿的完整生成流。"""

    name = "body_writing"
    description = "执行骨架规划、文本组装、引用校验及反思修订，输出正文终稿。"
    input_format = {
        "section": "Dict (包含 title, goal, word_count_target)",
        "fine_rag_context": "str (检索证据)",
        "available_citations": "Dict (可用引用库)",
        "task": "str",
        "prev_section_content": "str"
    }
    output_format = {
        "content": "str",
        "section_ref": "str"
    }

    def run(self, context: "PipelineContext", payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1. 骨架规划
        skeleton = SectionWritingService.plan_body_skeleton(context, payload)
        payload["skeleton"] = skeleton
        
        # 2. 文本组装
        content = SectionWritingService.compose_body_content(context, payload)
        
        # 3. 引用校验
        section = payload.get("section", {})
        available_citations = payload.get("available_citations", {})
        if hasattr(context, "_citation_validator"):
            available_citations, content = context._citation_validator.run_five_step_validation(
                available_citations, content, context.workspace.citations_validation_log, section.get("title", "正文")
            )
        
        # 4. 反思修订
        if hasattr(context, "_section_reflection_loop"):
            content = context._section_reflection_loop(content, section, available_citations)
        
        section_ref = context._get_section_ref(section) if hasattr(context, "_get_section_ref") else section.get("title", "正文")
        return {"content": str(content), "section_ref": str(section_ref)}

if __name__ == "__main__":
    sys.exit(run_component_cli(BodyWritingComponent()))