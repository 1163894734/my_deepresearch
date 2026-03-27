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
    from .section_writing_service import SectionWritingService
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli
    from section_writing_service import SectionWritingService


class IntroductionWritingComponent(JsonWorkflowComponent):
    """引言撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "引言", "goal": "交代背景与问题", "word_count_target": 600},
        "available_citations": {"Language Models are Few-Shot Learners": {"authors": "Brown et al.", "year": "2020", "title": "Language Models are Few-Shot Learners"}},
        "task": "（用户任务描述）",
        "prev_section_content": "（前一段内容）"
    }

    输出 JSON 示例:
    {
        "content": "引言正文..."
    }
    """

    name = "introduction_writing"
    description = "引言撰写组件：输入章节与上下文，输出通过反思后的引言终稿。"
    input_format = {
        "section": "Dict",
        "available_citations": "Dict",
        "task": "str",
        "prev_section_content": "str",
    }
    output_format = {
        "content": "str",
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

        final = SectionWritingService.write_intro_content_with_json(agent, payload)
        section = payload.get("section", {}) if isinstance(payload, dict) else {}
        section_ref = section.get("title", "引言") if isinstance(section, dict) else "引言"
        if hasattr(agent, "_log_reference_usage_in_section"):
            agent._log_reference_usage_in_section(str(section_ref), str(final))
        return {"content": str(final)}


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
    "input": {
        "section": {"title": "引言", "goal": "交代背景与问题", "word_count_target": 600},
        "available_citations": {
            "Large Language Models: A Survey": {
            "title": "Large Language Models: A Survey",
            "abstract": "本文综述了当前主流的大型语言模型，包括GPT、LLaMA和PaLM系列，并讨论了其特点、贡献与限制。同时概述了构建和增强LLM的技术，调查了用于训练、微调和评估LLM的数据集，回顾了广泛使用的评估指标，并在代表性基准上比较了多个流行LLM的性能。最后，讨论了开放挑战和未来研究方向。 Large Language Models (LLMs) have drawn a lot of attention due to their strong performance on a wide range of natural language tasks, since the release of ChatGPT in November 2022.",
            "keywords": [
                "Large Language Models",
                "GPT",
                "LLaMA",
                "PaLM",
                "ChatGPT"
            ],
            "authors": "Minaee, Shervin; Mikolov, Tomas; Nikzad, Narjes; Chenaghlu, Meysam; Socher, Richard; Amatriain, Xavier; Gao, Jianfeng",
            "url": "https://arxiv.org/abs/2402.06196",
            "published_time": "2024-02",
            "year": "2024",
            "source_type": "paper",
            "apa_citation": "(Minaee et al., 2024)"
            },
            "Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry": {
            "title": "Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry",
            "abstract": "本文总结了2024年大型语言模型黑客松的成果，展示了LLM在分子与材料属性预测、设计、自动化接口、科研沟通、数据管理、假设生成及文献知识提取等方面的多样化应用。活动在全球多地线上线下同步举行，共收到34个团队提交的作品，体现了LLM作为多功能机器学习模型和科研快速原型平台的双重价值。 These outcomes demonstrate the dual utility of LLMs as both multipurpose models for diverse machine learning tasks and platforms for rapid prototyping custom applications in scientific research.",
            "keywords": [
                "LLM",
                "材料科学",
                "分子属性预测",
                "科研自动化",
                "文献知识提取"
            ],
            "authors": "Various contributors from the LLM Hackathon community",
            "url": "https://arxiv.org/abs/2411.15221",
            "published_time": "2024-11",
            "year": "2024",
            "source_type": "paper",
            "apa_citation": "(LLM Hackathon Contributors, 2024)"
            }},
        "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
    }
}'''
        return run_component_cli(
            IntroductionWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(IntroductionWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
