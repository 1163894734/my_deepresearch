from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from ..long_writer_agent_v3 import LongWriterAgent
    except Exception:
        from long_writer_agent_v3 import LongWriterAgent

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .keyword_search_service import KeywordSearchPlanningService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from keyword_search_service import KeywordSearchPlanningService

from smolagents.monitoring import LogLevel

class KeywordSearchExpansionComponent(JsonWorkflowComponent):
    """关键词搜索与扩展（输入任务 -> 初始检索 -> 扩展检索）。"""

    name = "keyword_search_expansion"
    description = "关键词搜索及扩展搜索组件：从任务输入产出升级概念与两轮合并的检索文献。"
    input_format = {
        "task": "str, 清洗后的写作任务",
    }
    # 【修改输出格式】：适配 long_writer_agent_v3 要求的 available_citations 字段
    output_format = {
        "task": "str",
        "initial_concepts": "List[Dict]",
        "candidate_keywords": "List[str]",
        "upgraded_concepts": "List[Dict]",
        "available_citations": "Dict[str, Dict]" 
    }

    @staticmethod
    def _extract_initial_concepts(agent: "LongWriterAgent", task: str):
        return agent._keyword_search_service.extract_initial_concepts(agent, task)

    @staticmethod
    def _first_shot_retrieval(agent: "LongWriterAgent", concepts):
        return agent._keyword_search_service.first_shot_retrieval(agent, concepts)

    @staticmethod
    def _extract_and_filter_keywords(agent: "LongWriterAgent", papers, initial_concepts):
        return agent._keyword_search_service.extract_and_filter_keywords(agent, papers, initial_concepts)

    @staticmethod
    def _upgrade_concepts_with_keywords(agent: "LongWriterAgent", initial_concepts, candidate_words):
        return agent._keyword_search_service.upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words)

    @staticmethod
    def _second_shot_retrieval(agent: "LongWriterAgent", upgraded_concepts):
        return agent._keyword_search_service.second_shot_retrieval(agent, upgraded_concepts)

    def run(self, agent: "LongWriterAgent", payload: dict) -> dict:
        task = str(payload.get("task", "")).strip()

        agent.logger.log("📋 执行5步关键词搜索扩展流程...", level=LogLevel.INFO)
        
        initial_concepts = self._extract_initial_concepts(agent, task)
        first_shot_papers = self._first_shot_retrieval(agent, initial_concepts)
        candidate_keywords = self._extract_and_filter_keywords(agent, first_shot_papers, initial_concepts)
        upgraded_concepts = self._upgrade_concepts_with_keywords(agent, initial_concepts, candidate_keywords)
        second_shot_papers = self._second_shot_retrieval(agent, upgraded_concepts)

        # 【核心新增】：将第一轮和第二轮搜索到的论文列表合并到一个统一的大字典中
        combined_citations = {}
        for paper_dict in first_shot_papers:
            combined_citations.update(paper_dict)
        for paper_dict in second_shot_papers:
            combined_citations.update(paper_dict)

        agent.logger.log(f"✅ 5步搜索流程完成，两轮共收集并去重文献 {len(combined_citations)} 篇", level=LogLevel.INFO)

        return {
            "task": task,
            "initial_concepts": initial_concepts,
            "candidate_keywords": candidate_keywords,
            "upgraded_concepts": upgraded_concepts,
            "available_citations": combined_citations,  # 将合并后的文献字典作为核心输出
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