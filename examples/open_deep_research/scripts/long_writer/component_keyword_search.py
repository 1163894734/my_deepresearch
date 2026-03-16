from __future__ import annotations

import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .keyword_search_service import KeywordSearchPlanningService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from keyword_search_service import KeywordSearchPlanningService


# 输入 JSON 示例:
# {
#   "task": "大模型领域前沿进展"
# }
#
# 输出 JSON 示例:
# {
#   "task": "大模型领域前沿进展",
#   "initial_concepts": [{"concept_name": "大模型", "keywords": ["LLM", "Transformer"]}],
#   "top_20_papers": [{"title": "A Survey of Large Language Models", "abstract": "..."}],
#   "candidate_keywords": ["推理优化", "模型压缩"],
#   "upgraded_concepts": [{"concept_name": "大模型", "keywords": ["LLM", "推理优化"]}],
#   "final_retrieval_results": "..."
# }


class KeywordSearchExpansionComponent(JsonWorkflowComponent):
    """关键词搜索与扩展（输入任务 -> 初始检索 -> 扩展检索）。"""

    name = "keyword_search_expansion"
    description = "关键词搜索及扩展搜索组件：从任务输入产出升级概念与 Second-Shot 检索结果。"
    input_format = {
        "task": "str, 清洗后的写作任务",
    }
    output_format = {
        "task": "str",
        "initial_concepts": "List[Dict]",
        "top_20_papers": "List[Dict]",
        "candidate_keywords": "List[str]",
        "upgraded_concepts": "List[Dict]",
        "final_retrieval_results": "str",
    }

    @staticmethod
    def _extract_initial_concepts(agent, task: str):
        return agent._keyword_search_service.extract_initial_concepts(agent, task)

    @staticmethod
    def _first_shot_retrieval(agent, concepts):
        return agent._keyword_search_service.first_shot_retrieval(agent, concepts)

    @staticmethod
    def _parse_search_results_to_papers(text: str):
        return KeywordSearchPlanningService.parse_search_results_to_papers(text)

    @staticmethod
    def _extract_and_filter_keywords(agent, papers, initial_concepts):
        return agent._keyword_search_service.extract_and_filter_keywords(agent, papers, initial_concepts)

    @staticmethod
    def _upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words):
        return agent._keyword_search_service.upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words)

    @staticmethod
    def _second_shot_retrieval(agent, upgraded_concepts):
        return agent._keyword_search_service.second_shot_retrieval(agent, upgraded_concepts)

    def run(self, agent, payload: dict) -> dict:
        task = str(payload.get("task", "")).strip()

        initial_concepts = self._extract_initial_concepts(agent, task)
        top_20_papers = self._first_shot_retrieval(agent, initial_concepts)
        candidate_keywords = self._extract_and_filter_keywords(agent, top_20_papers, initial_concepts)
        upgraded_concepts = self._upgrade_concepts_with_keywords(agent, initial_concepts, candidate_keywords)
        final_retrieval_results = self._second_shot_retrieval(agent, upgraded_concepts)

        return {
            "task": task,
            "initial_concepts": initial_concepts,
            "top_20_papers": top_20_papers,
            "candidate_keywords": candidate_keywords,
            "upgraded_concepts": upgraded_concepts,
            "final_retrieval_results": final_retrieval_results,
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
