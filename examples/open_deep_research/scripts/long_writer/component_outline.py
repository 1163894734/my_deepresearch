from __future__ import annotations

import sys,json
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    try:
        from ..long_writer_agent_v3 import LongWriterAgent
    except Exception:
        from long_writer_agent_v3 import LongWriterAgent

from smolagents.monitoring import LogLevel

try:
    from .base_component import JsonWorkflowComponent
    from .cli_debugger import run_component_cli
except ImportError:
    from base_component import JsonWorkflowComponent
    from cli_debugger import run_component_cli



# 输入 JSON 示例:
# {
#   "task": "大模型领域前沿进展",
#   "upgraded_concepts": [{"concept_name": "推理能力", "keywords": ["CoT", "Self-Consistency"]}],
#   "top_20_papers": [{"title": "A Survey of Large Language Models", "abstract": "..."}],
#   "candidate_keywords": ["推理优化", "长上下文"],
#   "final_retrieval_results": "..."
# }
#
# 输出 JSON 示例:
# {
#   "outline_input": "...",
#   "outline_v1": "...",
#   "final_outline": "..."
# }


class OutlineGenerationReflectionComponent(JsonWorkflowComponent):
    """大纲生成与反思修订组件。"""

    name = "outline_generation_reflection"
    description = "大纲生成和反思修订组件：输入检索证据，输出最终大纲。"
    input_format = {
        "task": "str",
        "available_citations": "Dict{str, Dict}"
    }
    output_format = {
        "outline_input": "str",
        "outline_v1": "str",
        "final_outline": "str",
    }


    def run(self, agent: "LongWriterAgent", payload: dict) -> dict:
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


        outline_v1 = str(agent.execute_tool_call("outline_generation", {"input": json.dumps(payload, ensure_ascii=False)}))
        final_outline = self._outline_reflection_loop_v2(
            agent,
            outline_v1,
            payload
        )

        return {
            "outline_input": payload,
            "outline_v1": outline_v1,
            "final_outline": final_outline,
        }

    @staticmethod
    def _outline_reflection_loop_v2(
        agent: "LongWriterAgent",
        outline: str,
        payload: dict
    ) -> str:
        """
        主要作用：执行大纲反思与修订循环。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - outline (str): 待解析或待修订的大纲文本。
        - upgraded_concepts (List[Dict[str, Any]]): 融合扩展关键词后的升级概念集合。
        - retrieval_results (str): 该参数用于承载 `retrieval_results` 相关的业务上下文或控制信息。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        current = outline

        for i in range(agent.outline_max_iter):
            agent.logger.log(f"  大纲反思 {i+1}/{agent.outline_max_iter}", level=LogLevel.DEBUG)

            try:
                reflection_input = f"大纲:\n{current}\n参考资料:\n{payload}"
                report = str(agent.execute_tool_call("outline_reflection", {"input": reflection_input}))
                data = agent._parse_json(report)
                if not isinstance(data, dict):
                    data = {}

                score = data.get("score", 0)
                is_pass = data.get("is_pass", False)

                agent._log_reflection_to_file(f"大纲反思第{i+1}轮", score, report)

                if is_pass or score > agent.outline_score_threshold:
                    agent.logger.log(f"  ✅ 得分 {score}，大纲通过", level=LogLevel.DEBUG)
                    return current

                agent.logger.log(
                    f"  ⚠️ 得分 {score}，未达到通过标准（{agent.outline_score_threshold}），需要修订",
                    level=LogLevel.DEBUG,
                )

                revision_prompt = f"""大纲:\n{current}\n评审报告:\n{report}\n参考资料:\n{payload}"""
                current = str(agent.execute_tool_call("outline_revision", {"input": revision_prompt}))
                agent._log_revision_to_file(f"大纲修订第{i+1}轮", current)

            except Exception as e:
                agent.logger.log(f"  ⚠️ 反思出错: {e}", level=LogLevel.INFO)
                return current

        return current


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
    "query": "大模型领域前沿进展",
    "available_citations": {
        "OPUS: Optimizer-induced Projected Utility Selection": {
            "authors": "阿里巴巴, 上海交大, UW-Madison等",
            "year": "2026",
            "title": "OPUS: Optimizer-induced Projected Utility Selection for Dynamic Data Selection in LLM Pre-training",
            "url": "https://arxiv.org/pdf/2602.0540",
            "source_type": "paper",
            "apa_citation": "(阿里巴巴等, 2026)",
            "key_words": ["动态数据选择", "预训练优化", "AdamW", "Muon", "效用最大化", "Bench-Proxy"]
        },
        "Understanding Transformer Architecture through Continuous Dynamics: A Partial Differential Equation Perspective": {
            "authors": "Yukun Zhang et al.",
            "year": "2024",
            "title": "Understanding Transformer Architecture through Continuous Dynamics: A Partial Differential Equation Perspective",
            "url": "https://arxiv.org/abs/2408.09523",
            "source_type": "paper",
            "apa_citation": "(Zhang et al., 2024)",
            "key_words": ["Transformer", "偏微分方程", "连续动力学", "残差连接", "层归一化", "表示漂移"]
        }
    }
}'''

        return run_component_cli(
            OutlineGenerationReflectionComponent(),
            argv=["--input-json", payload_json_text],
        )

    return run_component_cli(OutlineGenerationReflectionComponent())


if __name__ == "__main__":
    raise SystemExit(main())
