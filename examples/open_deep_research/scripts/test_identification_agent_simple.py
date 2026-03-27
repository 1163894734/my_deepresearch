from __future__ import annotations

import json
from collections import Counter

from smolagents.models import ChatMessage, MessageRole, Model

try:
    from examples.open_deep_research.scripts.identification_agent import (
            ExpertVote,
        IdentificationAgent,
        KeywordDomainClassifier,
        SimilarityThresholdClusterer,
    )
except ImportError:
    from identification_agent import (  # type: ignore
            ExpertVote,
        IdentificationAgent,
        KeywordDomainClassifier,
        SimilarityThresholdClusterer,
    )


class DummyModel(Model):
    """Minimal model to satisfy ToolCallingAgent initialization."""

    def generate(self, messages, **kwargs):  # noqa: D401
        del messages, kwargs
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


class AlwaysApproveExpert:
    """Minimal expert judge for deterministic smoke tests."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, cluster, domain):
        del cluster, domain
        return ExpertVote(expert_name=self.name, approve=True, reason="smoke test approval")


class ThemeAwareNarrativeGenerator:
    """Deterministic but less fake narrative generator for clearer test outputs."""

    def generate(self, summaries, domain):
        text = "\n".join(str(s) for s in summaries).lower()

        tokens = [t for t in __import__("re").split(r"[^a-z0-9]+", text) if len(t) >= 4]
        stop = {
            "this",
            "that",
            "with",
            "from",
            "into",
            "uses",
            "using",
            "large",
            "language",
            "model",
            "models",
        }
        keyword_counter = Counter(t for t in tokens if t not in stop)
        key_terms = [k for k, _ in keyword_counter.most_common(3)]
        key_focus = " / ".join(key_terms) if key_terms else "general-focus"

        keyword_groups = {
            "multimodal": ["multimodal", "vision", "image", "vision-language", "cross-modal"],
            "rag": ["retrieval", "rag", "grounded", "citation", "reranker", "enterprise qa"],
            "context": ["long-context", "long context", "memory", "compression", "long sequence"],
        }
        scores = {name: sum(text.count(k) for k in kws) for name, kws in keyword_groups.items()}
        theme = max(scores, key=scores.get) if any(scores.values()) else "rag"

        if theme == "multimodal":
            return {
                "technology_term": f"{domain}-multimodal-fusion",
                "technology_problem": f"跨模态语义对齐与证据一致性不足，导致复杂场景理解稳定性下降（关键词焦点: {key_focus}）。",
                "technology_method": f"采用视觉-语言联合表征、跨模态检索与上下文重排，提升多源信息融合质量（关键技术词: {key_focus}）。",
                "application_direction": "面向多模态问答、文档理解和智能体规划等复合任务。",
            }
        if theme == "rag":
            return {
                "technology_term": f"{domain}-retrieval-grounding",
                "technology_problem": f"参数化记忆时效性与可追溯性不足，易出现事实漂移与幻觉（关键词焦点: {key_focus}）。",
                "technology_method": f"构建检索增强链路（召回-重排-生成）并注入可验证证据，平衡准确率与时延（关键技术词: {key_focus}）。",
                "application_direction": "适用于企业知识问答、合规审阅和高可靠客服场景。",
            }
        if theme == "context":
            return {
                "technology_term": f"{domain}-context-memory-optimization",
                "technology_problem": f"长上下文推理中存在记忆衰减与关键信息覆盖不足（关键词焦点: {key_focus}）。",
                "technology_method": f"通过上下文压缩、记忆扩展与结构化提示策略提升长序列推理稳定性（关键技术词: {key_focus}）。",
                "application_direction": "用于长文档分析、复杂流程推理与策略生成。",
            }
        return {
            "technology_term": f"{domain}-general-cluster",
            "technology_problem": "该簇仍处于探索期，问题定义与评价指标尚未统一。",
            "technology_method": "以任务驱动的实验迭代为主，结合检索和推理链逐步优化。",
            "application_direction": "可作为前沿候选方向持续跟踪。",
        }


def build_docs() -> list[dict]:
    # Intentionally mix several themes so filtering/clustering behavior is visible.
    return [
        {
            "doc_id": "rag_2022_a",
            "timestamp": "2022-05-01",
            "title": "RAG for enterprise QA",
            "abstract": "Large language model retrieval pipeline for grounded QA in enterprise search.",
            "text": "LLM retrieval reranker transformer grounding evidence.",
            "metadata": {"theme": "RAG"},
        },
        {
            "doc_id": "rag_2023_a",
            "timestamp": "2023-07-12",
            "title": "Improving retrieval-augmented generation",
            "abstract": "Transformer-based retriever with better context fusion for LLM answers.",
            "text": "retrieval augmented generation llm transformer context fusion",
            "metadata": {"theme": "RAG"},
        },
        {
            "doc_id": "rag_2024_a",
            "timestamp": "2024-03-18",
            "title": "Grounded long-form generation with retrieval",
            "abstract": "Large language model uses retrieval and citation to reduce hallucination.",
            "text": "large language model retrieval citation grounded generation",
            "metadata": {"theme": "RAG"},
        },
        {
            "doc_id": "rag_2025_a",
            "timestamp": "2025-08-06",
            "title": "Low-latency RAG architecture",
            "abstract": "LLM retrieval architecture for online latency and quality trade-off.",
            "text": "llm retrieval latency quality online transformer",
            "metadata": {"theme": "RAG"},
        },
        {
            "doc_id": "mm_2023_a",
            "timestamp": "2023-04-10",
            "title": "Multimodal LLM alignment",
            "abstract": "Transformer alignment across text and image for multimodal large language model.",
            "text": "multimodal llm transformer alignment text image",
            "metadata": {"theme": "MultiModal"},
        },
        {
            "doc_id": "mm_2024_a",
            "timestamp": "2024-11-22",
            "title": "Vision-language retrieval for LLM agents",
            "abstract": "Retrieval and large language model integration for multimodal agent planning.",
            "text": "vision language retrieval llm agent planning",
            "metadata": {"theme": "MultiModal"},
        },
        {
            "doc_id": "mm_2026_a",
            "timestamp": "2026-02-14",
            "title": "Long-context multimodal transformer",
            "abstract": "Transformer memory extension for large language model multimodal reasoning.",
            "text": "long context transformer multimodal llm reasoning",
            "metadata": {"theme": "MultiModal"},
        },
        {
            "doc_id": "battery_2024_a",
            "timestamp": "2024-01-05",
            "title": "Solid-state battery electrolyte",
            "abstract": "Electrochemical stability and ion transport in ceramic electrolyte design.",
            "text": "battery electrolyte ion transport electrochemistry",
            "metadata": {"theme": "Battery"},
        },
        {
            "doc_id": "battery_2025_a",
            "timestamp": "2025-09-19",
            "title": "Anode interface chemistry",
            "abstract": "Interfacial impedance control for lithium metal anode cycling life.",
            "text": "lithium anode interface impedance cycling",
            "metadata": {"theme": "Battery"},
        },
        {
            "doc_id": "battery_2026_a",
            "timestamp": "2026-06-30",
            "title": "Cathode microstructure optimization",
            "abstract": "Cathode porosity and binder distribution for high energy density.",
            "text": "cathode porosity binder energy density",
            "metadata": {"theme": "Battery"},
        },
    ]


def guess_theme_from_doc_ids(doc_ids: list[str]) -> str:
    mapping = {
        "rag": "RAG",
        "mm": "MultiModal",
        "battery": "Battery",
    }
    buckets = Counter()
    for doc_id in doc_ids:
        prefix = str(doc_id).split("_", 1)[0].lower()
        buckets[mapping.get(prefix, "Unknown")] += 1
    return buckets.most_common(1)[0][0] if buckets else "Unknown"


def print_cluster_comparison_table(result_obj: dict) -> None:
    clusters = result_obj.get("clusters", [])
    headers = ["cluster_id", "size", "theme_guess", "burst", "cagr", "representative_doc_ids"]
    rows = []
    for cluster in clusters:
        trend = cluster.get("trend", {}) if isinstance(cluster, dict) else {}
        rep_ids = cluster.get("representative_doc_ids", [])
        rep_ids_text = ",".join(rep_ids[:4])
        if len(rep_ids) > 4:
            rep_ids_text += ",..."
        rows.append(
            [
                str(cluster.get("cluster_id", "")),
                str(cluster.get("size", "")),
                guess_theme_from_doc_ids(rep_ids),
                str(bool(trend.get("burst_detected", False))),
                f"{float(trend.get('cagr', 0.0)):.3f}",
                rep_ids_text,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    print("\n===== CLUSTER COMPARISON TABLE =====")
    print(sep)
    print(head)
    print(sep)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)
    print("===== END COMPARISON TABLE =====\n")


def print_readable_summary(payload: dict, result_obj: dict) -> None:
    docs = payload.get("documents", [])
    themes = Counter(str(d.get("metadata", {}).get("theme", "Unknown")) for d in docs if isinstance(d, dict))

    print("\n===== IDENTIFICATION READABLE SUMMARY =====")
    print("Input domain:", payload.get("domain"))
    print("Input docs:", len(docs), "| Theme distribution:", dict(themes))
    print("Kept docs:", result_obj.get("kept_docs"), "| Removed docs:", result_obj.get("removed_docs"))

    clusters = result_obj.get("clusters", [])
    print("Approved clusters:", len(clusters))
    for i, c in enumerate(clusters, 1):
        trend = c.get("trend", {}) if isinstance(c, dict) else {}
        print(f"  [{i}] id={c.get('cluster_id')} size={c.get('size')} term={c.get('technology_term')}")
        print(f"      method={c.get('technology_method')}")
        print(
            "      trend:",
            f"cagr={trend.get('cagr')}",
            f"burst={trend.get('burst_detected')}",
            f"yearly_count={trend.get('yearly_count')}",
        )
        print(f"      representative_doc_ids={c.get('representative_doc_ids')}")
    print("===== END READABLE SUMMARY =====\n")
    print_cluster_comparison_table(result_obj)


def main() -> int:
    agent = IdentificationAgent(
        model=DummyModel(),
        tools=[],
        verbosity_level=0,
        classifier=KeywordDomainClassifier(
            {
                "llm": ["llm", "large language model", "transformer", "retrieval"],
            }
        ),
        # Keep deterministic behavior while allowing multiple clusters for readability.
        clusterer=SimilarityThresholdClusterer(threshold=0.45),
        narrative_generator=ThemeAwareNarrativeGenerator(),
        experts=[AlwaysApproveExpert("smoke-expert")],
    )

    payload = {
        "domain": "llm",
        "documents": build_docs(),
        "min_domain_probability": 0.35,
        "top_k_per_cluster": 5,
        "minimum_votes_to_pass": 1,
    }

    result = agent.run(json.dumps(payload, ensure_ascii=False))
    result_obj = json.loads(result)

    print_readable_summary(payload, result_obj)

    print("\n===== IDENTIFICATION FINAL RESULT =====")
    print(json.dumps(result_obj, ensure_ascii=False, indent=2))
    print("===== END IDENTIFICATION RESULT =====\n")

    print("[identification] kept_docs:", result_obj.get("kept_docs"))
    print("[identification] clusters:", len(result_obj.get("clusters", [])))

    assert result_obj.get("kept_docs", 0) > 0, "No kept docs in smoke test"
    assert len(result_obj.get("clusters", [])) >= 1, "No approved clusters in smoke test"
    print("[identification] smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
