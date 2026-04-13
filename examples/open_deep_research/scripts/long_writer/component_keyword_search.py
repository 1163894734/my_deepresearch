from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 引入我们新的上下文对象进行类型提示
    from scripts.multi_agent.agent_context import PipelineContext

try:
    from .base_component import JsonWorkflowComponent
    from .cli_debugger import run_component_cli
    from .keyword_search_service import KeywordSearchPlanningService
except ImportError:
    from base_component import JsonWorkflowComponent
    from cli_debugger import run_component_cli
    from keyword_search_service import KeywordSearchPlanningService

class KeywordSearchExpansionComponent(JsonWorkflowComponent):
    """关键词搜索与扩展（输入任务 -> 初始检索 -> 扩展检索）。"""

    name = "keyword_search_expansion"
    description = "关键词搜索及扩展搜索组件：从任务输入产出升级概念与两轮合并的检索文献。"
    input_format = {"task": "str, 清洗后的写作任务"}
    output_format = {
        "task": "str",
        "initial_concepts": "List[Dict]",
        "candidate_keywords": "List[str]",
        "upgraded_concepts": "List[Dict]",
        "available_citations": "Dict[str, Dict]" 
    }

    # 🔥 核心解耦：签名改为 context，并直接调用底层 Service，删除了冗余的 staticmethod
    def run(self, context: "PipelineContext", payload: dict) -> dict:
        task = str(payload.get("task", "")).strip()
        
        # 将 context 作为运行环境传递给底层 Service
        initial_concepts = KeywordSearchPlanningService.extract_initial_concepts(context, task)
        first_shot_papers = KeywordSearchPlanningService.first_shot_retrieval(context, initial_concepts)
        candidate_keywords = KeywordSearchPlanningService.extract_and_filter_keywords(context, first_shot_papers, initial_concepts)
        upgraded_concepts = KeywordSearchPlanningService.upgrade_concepts_with_keywords(context, initial_concepts, candidate_keywords)
        second_shot_papers = KeywordSearchPlanningService.second_shot_retrieval(context, upgraded_concepts)

        combined_citations = {}
        for paper_dict in first_shot_papers:
            combined_citations.update(paper_dict)
        for paper_dict in second_shot_papers:
            combined_citations.update(paper_dict)

        return {
            "task": task,
            "initial_concepts": initial_concepts,
            "candidate_keywords": candidate_keywords,
            "upgraded_concepts": upgraded_concepts,
            "available_citations": combined_citations,
            "combined_citations_count": len(combined_citations)
        }

def main() -> int:
    if len(sys.argv) <= 1:
        payload_json_text = r'''{
    "task": "大模型领域前沿进展"
}'''
        return run_component_cli(
            KeywordSearchExpansionComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(KeywordSearchExpansionComponent())

if __name__ == "__main__":
    raise SystemExit(main())