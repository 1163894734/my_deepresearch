from __future__ import annotations

import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .citation_flow_service import CitationFlowService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from citation_flow_service import CitationFlowService


class CitationValidationComponent(JsonWorkflowComponent):
    """引用验证组件。

    输入 JSON 示例:
    {}

    输出 JSON 示例:
    {
        "validation_result": {
            "status": "completed",
            "total": 8,
            "verified": 6,
            "rejected": 2
        }
    }
    """

    name = "citation_validation"
    description = "引用验证组件：执行 Search→Verify→Retrieve→Validate→Add 的验证流程。"
    input_format = {}
    output_format = {
        "validation_result": "Dict",
    }

    def run(self, agent, payload: dict) -> dict:
        validation_result = CitationFlowService.validate_all_citations(agent)
        return {"validation_result": validation_result}


def main() -> int:
    if len(sys.argv) <= 1:
        payload_json_text = r'''{}'''
        return run_component_cli(
            CitationValidationComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(CitationValidationComponent())


if __name__ == "__main__":
    raise SystemExit(main())
