from __future__ import annotations

import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli


class ReferencesWritingComponent(JsonWorkflowComponent):
    """参考文献撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "参考文献"}
    }

    输出 JSON 示例:
    {
        "content": "[1] Author. (Year). Title."
    }
    """

    name = "references_writing"
    description = "参考文献撰写组件：输入章节配置，输出引用库格式化结果。"
    input_format = {
        "section": "Dict",
    }
    output_format = {
        "content": "str",
    }

    def run(self, agent, payload: dict) -> dict:
        section = payload.get("section", {})
        content = agent._write_references_section(section)
        return {"content": str(content)}


def main() -> int:
    if len(sys.argv) <= 1:
        payload_json_text = r'''{
    "section": {"title": "参考文献"}
}'''
        return run_component_cli(
            ReferencesWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(ReferencesWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
