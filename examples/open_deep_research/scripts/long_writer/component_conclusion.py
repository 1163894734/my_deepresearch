from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .section_writing_service import SectionWritingService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from section_writing_service import SectionWritingService


class ConclusionWritingComponent(JsonWorkflowComponent):
    """结论撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "结论", "goal": "总结贡献与展望", "word_count_target": 500},
        "fine_rag_context": "全文内容/总结材料...",
        "available_citations": {"Minaee et al. (2024)": {"authors": "Minaee et al.", "year": "2024"}}
    }

    输出 JSON 示例:
    {
        "content": "结论正文..."
    }
    """

    name = "conclusion_writing"
    description = "结论撰写组件：输入章节与上下文，输出通过反思后的结论终稿。"
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
        word_count_target = int(section.get("word_count_target", 0) or 0)

        conclusion_input = SectionWritingService.build_section_input(
            agent,
            section,
            "conclusion",
            fine_rag_context,
            available_citations,
        )
        content = agent.execute_tool_call("conclusion_write", {"input": conclusion_input})
        return {"content": str(content)}


def main() -> int:
    if len(sys.argv) <= 1:
        workspace_root = Path(__file__).resolve().parents[4]
        long_text_path = workspace_root / "long_text.md"
        fine_rag_context = long_text_path.read_text(encoding="utf-8") if long_text_path.exists() else ""

        payload_json_text = r'''{
    "input": {
        "section": {"title": "结论", "goal": "总结贡献与展望", "word_count_target": 500},
        "fine_rag_context": __FINE_RAG_CONTEXT__,
        "available_citations": {
            "Naveed et al. (2023)": {"authors": "Naveed et al.", "year": "2023", "title": "A Comprehensive Overview of Large Language Models"}
        }
    },
    "agent_config": {
        "overrides": {
            "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
        }
    }
}'''
        payload_json_text = payload_json_text.replace("__FINE_RAG_CONTEXT__", json.dumps(fine_rag_context, ensure_ascii=False))
        return run_component_cli(
            ConclusionWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(ConclusionWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
