from __future__ import annotations

import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .section_writing_service import SectionWritingService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from section_writing_service import SectionWritingService


class BodyWritingComponent(JsonWorkflowComponent):
    """普通段落撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "2.1 推理能力", "goal": "分析推理技术", "word_count_target": 900},
        "fine_rag_context": "检索证据...",
        "available_citations": {"Wei et al. (2022)": {"authors": "Wei et al.", "year": "2022"}}
    }

    输出 JSON 示例:
    {
        "content": "正文段落..."
    }
    """

    name = "body_writing"
    description = "普通段落撰写组件：输入章节与证据，输出通过反思后的正文终稿。"
    input_format = {
        "section": "Dict",
        "fine_rag_context": "str",
        "available_citations": "Dict",
    }
    output_format = {
        "content": "str",
    }

    def run(self, agent, payload: dict) -> dict:
        section = payload.get("section", {})
        fine_rag_context = str(payload.get("fine_rag_context", ""))
        available_citations = payload.get("available_citations", {})
        final = SectionWritingService.write_body_content(agent, section, fine_rag_context, available_citations)
        return {"content": str(final)}


def main() -> int:
    if len(sys.argv) <= 1:
        payload_json_text = r'''{
    "input": {
        "section": {"title": "2.1 推理优化机制", "goal": "分析OPRO机制与优势", "word_count_target": 900},
        "fine_rag_context": "OPRO 将优化过程转化为自然语言迭代过程。",
        "available_citations": {
            "Yang et al. (2023)": {"authors": "Yang et al.", "year": "2023", "title": "Large Language Models as Optimizers"}
        }
    },
    "agent_config": {
        "overrides": {
            "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
        }
    }
}'''
        return run_component_cli(
            BodyWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(BodyWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
