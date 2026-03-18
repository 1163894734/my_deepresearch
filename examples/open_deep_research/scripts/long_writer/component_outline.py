from __future__ import annotations

import sys
from typing import Any, Dict, List

from smolagents.monitoring import LogLevel

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli


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
        "upgraded_concepts": "List[Dict]",
        "top_20_papers": "List[Dict]",
        "candidate_keywords": "List[str]",
        "final_retrieval_results": "str",
    }
    output_format = {
        "outline_input": "str",
        "outline_v1": "str",
        "final_outline": "str",
    }

    @staticmethod
    def _build_outline_generation_input(
        agent,
        task: str,
        upgraded_concepts: List[Dict[str, Any]],
        top_20_papers: List[Dict[str, str]],
        candidate_keywords: List[str],
        final_retrieval_results: str,
    ) -> str:
        """
        主要作用：为大纲组件组装最终输入提示。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - task (str): 用户给出的原始写作任务，或经清洗后的任务描述。
        - upgraded_concepts (List[Dict[str, Any]]): 融合扩展关键词后的升级概念集合。
        - top_20_papers (List[Dict[str, str]]): 该参数用于承载 `top_20_papers` 相关的业务上下文或控制信息。
        - candidate_keywords (List[str]): 从检索材料中提炼得到的候选关键词列表。
        - final_retrieval_results (str): 该参数用于承载 `final_retrieval_results` 相关的业务上下文或控制信息。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前上下文中的关键字段。
        - 按既定模板和业务规则拼装输入结构。
        - 返回下游阶段可直接消费的提示词、映射或载荷。
        """

        return agent._keyword_search_service.build_outline_generation_input(
            agent,
            task,
            upgraded_concepts,
            top_20_papers,
            candidate_keywords,
            final_retrieval_results,
        )

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
        upgraded_concepts = payload.get("upgraded_concepts", [])
        top_20_papers = payload.get("top_20_papers", [])
        candidate_keywords = payload.get("candidate_keywords", [])
        final_retrieval_results = str(payload.get("final_retrieval_results", ""))

        outline_input = self._build_outline_generation_input(
            agent,
            task=task,
            upgraded_concepts=upgraded_concepts,
            top_20_papers=top_20_papers,
            candidate_keywords=candidate_keywords,
            final_retrieval_results=final_retrieval_results,
        )
        outline_v1 = str(agent.execute_tool_call("outline_generation", {"input": outline_input}))
        final_outline = self._outline_reflection_loop_v2(
            agent,
            outline_v1,
            upgraded_concepts,
            final_retrieval_results,
        )

        return {
            "outline_input": outline_input,
            "outline_v1": outline_v1,
            "final_outline": final_outline,
        }

    @staticmethod
    def _outline_reflection_loop_v2(
        agent,
        outline: str,
        upgraded_concepts: List[Dict[str, Any]],
        retrieval_results: str,
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
                reflection_input = f"大纲:\n{current}"
                reflection_input += "\n\n【核心概念组】需要确保大纲覆盖以下所有概念：\n"
                for idx, concept in enumerate(upgraded_concepts, 1):
                    name = concept.get("concept_name", "")
                    kws = concept.get("keywords", [])
                    reflection_input += f"{idx}. {name}: {', '.join(kws)}\n"
                reflection_input += "\n请检查大纲是否覆盖了所有概念组的核心内容。"

                if retrieval_results:
                    reflection_input += f"\n\n【Second-Shot 精准检索结果】\n{retrieval_results[:3000]}\n"
                    reflection_input += "\n请确保大纲的各章节能够充分利用这些检索到的内容。"

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

                revision_prompt = f"""大纲:
{current}

评审报告:
{report}

【参考：核心概念组】
"""
                for concept in upgraded_concepts:
                    name = concept.get("concept_name", "")
                    kws = concept.get("keywords", [])
                    revision_prompt += f"- {name}: {', '.join(kws)}\n"

                if retrieval_results:
                    revision_prompt += f"\n【参考：Second-Shot 检索结果】\n{retrieval_results[:2000]}\n"

                revision_prompt += """

重要提示：请直接输出修改后的完整大纲，不要使用任何修改标记（如~~删除线~~、**加粗**等），只输出最终的干净文本。确保大纲覆盖所有概念组的内容。"""

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
    "task": "大模型领域前沿进展",
    "upgraded_concepts": [
        {
            "concept_name": "推理优化",
            "keywords": ["OPRO", "自然语言优化", "提示工程自动化", "无梯度优化"]
        },
        {
            "concept_name": "大模型综述",
            "keywords": ["LoRA", "前缀微调", "指令微调", "RAG"]
        }
    ],
    "top_20_papers": [
        {
            "title": "Large Language Models as Optimizers",
            "abstract": "该研究提出OPRO框架，利用大语言模型作为优化器，通过自然语言迭代优化任务目标。",
            "keywords": ["OPRO优化框架", "自然语言优化循环", "提示工程自动化"],
            "authors": "Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen",
            "url": "https://arxiv.org/abs/2309.03409",
            "published_time": "2023-09",
            "year": "2023",
            "source_type": "paper",
            "apa_citation": "(Yang et al., 2023)"
        },
        {
            "title": "A Comprehensive Overview of Large Language Models",
            "abstract": "该综述总结了大模型在架构创新、训练策略、微调技术与推理优化方面的进展。",
            "keywords": ["LoRA低秩适配", "前缀微调技术", "指令微调范式", "检索增强生成(RAG)"],
            "authors": "Humza Naveed, Asad Ullah Khan, Qiu Yuanchao, Muhammad Ali Raza, Farhan Hassan Khan",
            "url": "https://arxiv.org/abs/2307.06435",
            "published_time": "2023-07",
            "year": "2023",
            "source_type": "paper",
            "apa_citation": "(Naveed et al., 2023)"
        }
    ],
    "candidate_keywords": ["推理优化", "OPRO", "LoRA", "指令微调", "RAG"],
    "final_retrieval_results": "[1] Large Language Models as Optimizers\\nURL: https://arxiv.org/abs/2309.03409\\n时间: 2023-09\\n摘要: 该研究提出OPRO框架，利用大语言模型作为优化器，通过自然语言迭代优化任务目标。\\n关键词: OPRO优化框架, 自然语言优化循环, 提示工程自动化\\n\\n[2] A Comprehensive Overview of Large Language Models\\nURL: https://arxiv.org/abs/2307.06435\\n时间: 2023-07\\n摘要: 该综述总结了大模型在架构创新、训练策略、微调技术与推理优化方面的进展。\\n关键词: LoRA低秩适配, 前缀微调技术, 指令微调范式, 检索增强生成(RAG)"
}'''

        return run_component_cli(
            OutlineGenerationReflectionComponent(),
            argv=["--input-json", payload_json_text],
        )

    return run_component_cli(OutlineGenerationReflectionComponent())


if __name__ == "__main__":
    raise SystemExit(main())
