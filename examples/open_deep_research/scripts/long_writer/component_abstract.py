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


class AbstractWritingComponent(JsonWorkflowComponent):
    """摘要撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "摘要", "goal": "凝练核心发现", "word_count_target": 300},
        "conclusion_text": "本文提出..."
    }

    输出 JSON 示例:
    {
        "content": "摘要正文..."
    }
    """

    name = "abstract_writing"
    description = "摘要撰写组件：输入摘要章节配置与结论文本，输出摘要终稿。"
    input_format = {
        "section": "Dict",
        "conclusion_text": "str",
    }
    output_format = {
        "content": "str",
    }

    def run(self, agent, payload: dict) -> dict:
        section = payload.get("section", {})
        conclusion_text = str(payload.get("conclusion_text", ""))
        content = SectionWritingService.write_abstract_section(agent, section, conclusion_text)
        return {"content": str(content)}


def main() -> int:
    if len(sys.argv) <= 1:
        workspace_root = Path(__file__).resolve().parents[4]
        long_text_path = workspace_root / "long_text.md"
        fine_rag_context = long_text_path.read_text(encoding="utf-8") if long_text_path.exists() else ""
        payload_json_text = r'''{
    "input": {
        "section": {"title": "摘要", "goal": "凝练核心发现", "word_count_target": 300},
        "fine_rag_context": __FINE_RAG_CONTEXT__,
        "conclusion_text": "本文围绕大模型推理优化与参数高效微调展开，指出未来方向。"
    },
    "agent_config": {
        "overrides": {
            "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
        }
    }
}'''
        payload_json_text = payload_json_text.replace("__FINE_RAG_CONTEXT__", json.dumps(fine_rag_context, ensure_ascii=False))
        return run_component_cli(
            AbstractWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(AbstractWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
