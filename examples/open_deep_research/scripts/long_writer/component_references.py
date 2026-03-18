from __future__ import annotations

import re
import sys

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli


class ReferencesWritingComponent(JsonWorkflowComponent):
    """参考文献撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "参考文献"},
        "available_citations": {"A Comprehensive Overview of Large Language Models": {"authors": "Naveed et al.", "year": "2023", "title": "A Comprehensive Overview of Large Language Models"}},
        "task": "（用户任务描述）"
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
        "available_citations": "Dict",
        "task": "str",
    }
    output_format = {
        "content": "str",
    }

    @staticmethod
    def _normalize_available_citations(raw) -> dict:
        """
        主要作用：规范化组件输入中的引用字典。

        输入参数：
        - raw: 未经规范化的原始输入，可为字典、列表、字符串或混合结构。

        返回值：
        - dict：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """

        if isinstance(raw, dict):
            normalized = {}
            for k, v in raw.items():
                info = dict(v) if isinstance(v, dict) else {"title": str(v)}
                title = str(info.get("title") or k).strip()
                if not title:
                    continue
                info["title"] = title
                authors = str(info.get("authors", "")).strip()
                year = str(info.get("year", "")).strip()
                if authors and year:
                    info.setdefault("canonical_key", f"{authors} ({year})")
                normalized[title] = info
            return normalized

        if isinstance(raw, list):
            normalized = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                apa = str(item.get("apa_citation", "")).strip()
                authors = str(item.get("authors", "")).strip()
                year = str(item.get("year", "")).strip()
                title = str(item.get("title", "")).strip()
                if not title and apa:
                    title = apa
                if not title:
                    continue
                info = dict(item)
                info["title"] = title
                if authors and year:
                    info.setdefault("canonical_key", f"{authors} ({year})")
                normalized[title] = info
            return normalized

        return {}

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

        payload_obj = payload.get("input") if isinstance(payload, dict) and isinstance(payload.get("input"), dict) else payload
        payload_obj = payload_obj if isinstance(payload_obj, dict) else {}

        section = payload_obj.get("section", {})

        # main/CLI 测试时：将 available_citations 直接写入 agent._citations 再输出
        normalized_citations = self._normalize_available_citations(payload_obj.get("available_citations"))
        if normalized_citations:
            if not isinstance(getattr(agent, "_citations", None), dict):
                agent._citations = {}
            for _title, info in normalized_citations.items():
                key = str(info.get("canonical_key") or "").strip()
                if not key:
                    authors = str(info.get("authors", "")).strip()
                    year = str(info.get("year", "")).strip()
                    key = f"{authors} ({year})" if authors and year else str(info.get("title") or "").strip()
                if not key:
                    continue
                if key not in agent._citations:
                    agent._citation_counter = int(getattr(agent, "_citation_counter", 0) or 0) + 1
                    merged = dict(info)
                    merged.setdefault("id", agent._citation_counter)
                    agent._citations[key] = merged
                else:
                    existing = dict(agent._citations.get(key, {}))
                    existing.update(dict(info))
                    agent._citations[key] = existing

        content = agent._write_references_section(section)
        return {"content": str(content)}


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
        "section": {"title": "参考文献"},
        "available_citations": [{
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
            {
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
            }],
        "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
    }
}'''
        return run_component_cli(
            ReferencesWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(ReferencesWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
