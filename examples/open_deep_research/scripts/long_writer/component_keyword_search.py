from __future__ import annotations

import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
    from .keyword_search_service import KeywordSearchPlanningService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from keyword_search_service import KeywordSearchPlanningService

from smolagents.monitoring import LogLevel


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
        """
        主要作用：调用关键词服务抽取初始概念。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - task (str): 用户给出的原始写作任务，或经清洗后的任务描述。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """

        return agent._keyword_search_service.extract_initial_concepts(agent, task)

    @staticmethod
    def _first_shot_retrieval(agent, concepts):
        """
        主要作用：调用关键词服务执行第一轮检索。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - concepts: 概念集合，通常包含 concept_name 与 keywords。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

        return agent._keyword_search_service.first_shot_retrieval(agent, concepts)

    @staticmethod
    def _parse_search_results_to_papers(text: str):
        """
        主要作用：解析检索输出为论文列表。

        输入参数：
        - text (str): 待解析、清洗或重写的原始文本内容。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

        return KeywordSearchPlanningService.parse_search_results_to_papers(text)

    @staticmethod
    def _extract_and_filter_keywords(agent, papers, initial_concepts):
        """
        主要作用：调用关键词服务提取并筛选候选关键词。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - papers: 论文材料列表。
        - initial_concepts: 任务初步抽取得到的概念集合。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """

        return agent._keyword_search_service.extract_and_filter_keywords(agent, papers, initial_concepts)

    @staticmethod
    def _upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words):
        """
        主要作用：调用关键词服务生成升级概念。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - initial_concepts: 任务初步抽取得到的概念集合。
        - candidate_words: 准备并入概念集合的候选关键词。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        return agent._keyword_search_service.upgrade_concepts_with_keywords(agent, initial_concepts, candidate_words)

    @staticmethod
    def _second_shot_retrieval(agent, upgraded_concepts):
        """
        主要作用：调用关键词服务执行第二轮检索。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - upgraded_concepts: 融合扩展关键词后的升级概念集合。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

        return agent._keyword_search_service.second_shot_retrieval(agent, upgraded_concepts)

    def run(self, agent, payload: dict) -> dict:
        """
        主要作用：执行当前组件的主流程，处理输入并返回结构化输出。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - dict：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        task = str(payload.get("task", "")).strip()

        agent.logger.log("📋 执行5步关键词搜索扩展流程...", level=LogLevel.INFO)
        
        initial_concepts = self._extract_initial_concepts(agent, task)
        top_20_papers = self._first_shot_retrieval(agent, initial_concepts)
        candidate_keywords = self._extract_and_filter_keywords(agent, top_20_papers, initial_concepts)
        upgraded_concepts = self._upgrade_concepts_with_keywords(agent, initial_concepts, candidate_keywords)
        final_retrieval_results = self._second_shot_retrieval(agent, upgraded_concepts)

        agent.logger.log("✅ 5步关键词搜索扩展流程完成", level=LogLevel.INFO)

        return {
            "task": task,
            "initial_concepts": initial_concepts,
            "top_20_papers": top_20_papers,
            "candidate_keywords": candidate_keywords,
            "upgraded_concepts": upgraded_concepts,
            "final_retrieval_results": final_retrieval_results,
        }


def main() -> int:
    """
    主要作用：执行 main 相关逻辑。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """

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
