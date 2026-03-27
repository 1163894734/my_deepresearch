from __future__ import annotations

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.tools import Tool

try:
    from examples.open_deep_research.scripts.interpretation_agent import InterpretationAgent
except ImportError:
    from interpretation_agent import InterpretationAgent  # type: ignore


class DummyInterpretationModel(Model):
    """Rule-based fake model for deterministic offline smoke tests."""

    def generate(self, messages, **kwargs):  # noqa: D401
        del kwargs
        prompt = str(messages[-1].content[0].get("text", "")) if messages else ""

        if "提取一个最核心的技术名词" in prompt:
            content = "检索增强生成"
        elif "你是裁判编辑" in prompt and "只输出 JSON" in prompt:
            content = '{"pass": true, "issues": [], "revision_instruction": ""}'
        elif "你是技术事实整理员" in prompt:
            content = (
                "检索增强生成是一种将外部知识检索与生成模型结合的方法。"
                "其关键环节包括检索器召回、上下文拼接与生成器回答。"
                "该方法可提升时效性和可追溯性，但受检索质量与上下文长度限制。"
            )
        elif "请基于以下客观事实材料" in prompt:
            if "《辞海》编辑" in prompt:
                content = (
                    "检索增强生成是将信息检索与文本生成耦合的技术路径，"
                    "通常由检索器、重排模块与生成器组成。"
                    "其目标是在回答阶段引入外部证据，以提高事实一致性与可追溯性。"
                )
            elif "智库分析师" in prompt:
                content = (
                    "从产业落地看，检索增强生成已在企业知识问答、客服辅助和合规审阅中形成可复制场景。"
                    "其价值在于用可更新知识库弥补参数化记忆时滞，但系统瓶颈集中在召回精度、索引更新频率和在线时延三项指标。"
                    "短期竞争关键将转向检索评测体系与端到端成本控制能力。"
                )
            elif "科普大V" in prompt:
                content = (
                    "你可以把检索增强生成理解成“开卷考试”的AI："
                    "先去资料库翻书，再组织答案，而不是只靠脑子硬想。"
                    "这样做能减少一本正经胡说八道，但如果翻到的资料不准、过期，最后答案也会被带偏。"
                )
            else:
                content = (
                    "检索增强生成通过外部知识注入降低模型幻觉风险。"
                    "在问答、企业知识库与客服场景中应用广泛，"
                    "但仍需在召回精度和延迟之间取得平衡。"
                )
        else:
            content = "默认响应"

        return ChatMessage(role=MessageRole.ASSISTANT, content=content)


class DummyCoarseRAGTool(Tool):
    name = "coarse_rag"
    description = "Return local retrieval snippets for testing"
    inputs = {"input": {"type": "string", "description": "query"}}
    output_type = "string"

    def forward(self, input):
        del input
        return (
            "RAG combines retrieval and generation.\n\n"
            "Retriever fetches relevant passages from a corpus.\n\n"
            "Generator conditions on retrieved passages to answer."
        )


class DummyWebSearchTool(Tool):
    name = "web_search"
    description = "Return web snippets for testing"
    inputs = {"query": {"type": "string", "description": "query"}}
    output_type = "string"

    def forward(self, query):
        del query
        return (
            "RAG improves grounding in production QA systems.\n\n"
            "Typical risks include stale indexes and low-quality retrieval."
        )


def main() -> int:
    agent = InterpretationAgent(
        model=DummyInterpretationModel(),
        tools=[DummyCoarseRAGTool(), DummyWebSearchTool()],
    )

    result = agent.run("请解读检索增强生成（RAG）的技术原理、应用和局限")
    print("\n===== INTERPRETATION FINAL RESULT =====")
    print(result)
    print("===== END INTERPRETATION RESULT =====\n")

    assert "技术名词解读" in result, "Missing output title"
    assert "检索增强生成" in result, "Missing extracted term"
    assert "客观事实底稿" in result, "Missing facts section"
    print("[interpretation] smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
