from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from smolagents import ToolCallingAgent
from smolagents.agents import ActionOutput, ToolOutput
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel


# =========================
# Data Models
# =========================


@dataclass
class TimestampedDocument:
	"""输入文档对象。"""

	doc_id: str
	timestamp: str
	text: str = ""
	title: str = ""
	abstract: str = ""
	introduction_or_conclusion: str = ""
	independent_claim: str = ""
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainScoredDocument:
	"""带领域置信度的文档。"""

	document: TimestampedDocument
	core_text: str
	domain_probability: float


@dataclass
class TrendMetrics:
	"""技术簇时间演化指标。"""

	yearly_count: Dict[int, int]
	yearly_share: Dict[int, float]
	cagr: float
	burst_detected: bool
	burst_reason: str


@dataclass
class ExpertVote:
	"""单个专家评审结果。"""

	expert_name: str
	approve: bool
	reason: str


@dataclass
class ClusterInsight:
	"""单个聚类簇洞察。"""

	cluster_id: int
	size: int
	representative_doc_ids: List[str]
	representative_summaries: List[str]
	technology_term: str
	technology_problem: str
	technology_method: str
	application_direction: str
	trend: TrendMetrics
	votes: List[ExpertVote]
	approved: bool


@dataclass
class FrontierIdentificationReport:
	"""最终输出报告对象。"""

	domain: str
	total_input_docs: int
	kept_docs: int
	removed_docs: int
	clusters: List[ClusterInsight]


# =========================
# Protocols / Interfaces
# =========================


class DomainClassifier(Protocol):
	"""领域判别器接口。"""

	def predict_domain_probability(self, texts: Sequence[str], domain: str) -> List[float]:
		...


class TextEmbedder(Protocol):
	"""向量化接口。"""

	def embed(self, texts: Sequence[str]) -> List[List[float]]:
		...


class Clusterer(Protocol):
	"""聚类接口。"""

	def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
		...


class ClusterNarrativeGenerator(Protocol):
	"""技术簇内涵抽取接口。"""

	def generate(self, summaries: Sequence[str], domain: str) -> Dict[str, str]:
		...


class ExpertJudge(Protocol):
	"""专家评审接口。"""

	def evaluate(self, cluster: ClusterInsight, domain: str) -> ExpertVote:
		...


# =========================
# Core Components
# =========================


class CoreTextExtractor:
	"""文本瘦身器：标题 + 摘要 + 权利要求/引言结论。"""

	def build_core_text(self, doc: TimestampedDocument, max_chars: int = 1400) -> str:
		parts: List[str] = []
		if doc.title:
			parts.append(f"[标题] {doc.title.strip()}")
		if doc.abstract:
			parts.append(f"[摘要] {doc.abstract.strip()}")
		if doc.independent_claim:
			parts.append(f"[独立权利要求] {doc.independent_claim.strip()}")
		if doc.introduction_or_conclusion:
			parts.append(f"[引言/结论] {doc.introduction_or_conclusion.strip()}")

		if not parts and doc.text:
			parts.append(doc.text.strip())

		merged = "\n".join(parts).strip()
		if len(merged) <= max_chars:
			return merged
		return merged[:max_chars] + "..."


class KeywordDomainClassifier:
	"""轻量领域二分类器（规则基线，可替换为微调模型）。"""

	def __init__(self, domain_keywords: Optional[Dict[str, List[str]]] = None):
		self.domain_keywords = domain_keywords or {}

	def predict_domain_probability(self, texts: Sequence[str], domain: str) -> List[float]:
		keywords = [k.lower() for k in self.domain_keywords.get(domain, [])]
		if not keywords:
			return [0.5 for _ in texts]

		probs: List[float] = []
		for text in texts:
			t = text.lower()
			hits = sum(1 for k in keywords if k in t)
			# 平滑映射到 (0,1)
			prob = 1.0 - math.exp(-hits / max(1.0, len(keywords) * 0.35))
			probs.append(round(min(max(prob, 0.0), 1.0), 6))
		return probs


class SklearnRandomForestDomainClassifier:
	"""可选：随机森林二分类器（需先 fit）。"""

	def __init__(self):
		self._ready = False
		self._vectorizer = None
		self._clf = None

	def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
		try:
			from sklearn.ensemble import RandomForestClassifier
			from sklearn.feature_extraction.text import TfidfVectorizer
		except Exception as exc:  # pragma: no cover - 依赖缺失时仅运行时报错
			raise ImportError("需要安装 scikit-learn 才能使用 SklearnRandomForestDomainClassifier") from exc

		self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
		x = self._vectorizer.fit_transform(texts)
		self._clf = RandomForestClassifier(
			n_estimators=300,
			random_state=42,
			n_jobs=-1,
			class_weight="balanced_subsample",
		)
		self._clf.fit(x, labels)
		self._ready = True

	def predict_domain_probability(self, texts: Sequence[str], domain: str) -> List[float]:
		del domain
		if not self._ready:
			raise RuntimeError("SklearnRandomForestDomainClassifier 尚未 fit")
		x = self._vectorizer.transform(texts)
		probs = self._clf.predict_proba(x)[:, 1]
		return [float(p) for p in probs]


class HashingTextEmbedder:
	"""默认无依赖向量器（可替换 BGE-M3）。"""

	def __init__(self, dim: int = 256):
		self.dim = dim

	def _tokenize(self, text: str) -> List[str]:
		return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", text.lower()) if t]

	def embed(self, texts: Sequence[str]) -> List[List[float]]:
		vectors: List[List[float]] = []
		for text in texts:
			v = [0.0] * self.dim
			for tok in self._tokenize(text):
				idx = hash(tok) % self.dim
				v[idx] += 1.0
			# L2 normalize
			norm = math.sqrt(sum(x * x for x in v)) or 1.0
			vectors.append([x / norm for x in v])
		return vectors


class BGEM3Embedder:
	"""可选：BGE-M3 向量器（sentence-transformers + GPU）。"""

	def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 128):
		self.model_name = model_name
		self.batch_size = batch_size
		self._model = None

	def _ensure_model(self) -> None:
		if self._model is not None:
			return
		try:
			from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
		except Exception as exc:  # pragma: no cover
			raise ImportError("需要安装 sentence-transformers 才能使用 BGEM3Embedder") from exc
		self._model = SentenceTransformer(self.model_name)

	def embed(self, texts: Sequence[str]) -> List[List[float]]:
		self._ensure_model()
		arr = self._model.encode(
			list(texts),
			batch_size=self.batch_size,
			show_progress_bar=False,
			normalize_embeddings=True,
		)
		return [list(map(float, row)) for row in arr]


class UmapHdbscanClusterer:
	"""UMAP 降维 + HDBSCAN 聚类；缺依赖时自动回退。"""

	def __init__(self, umap_dim: int = 10, min_cluster_size: int = 8, fallback_similarity: float = 0.78):
		self.umap_dim = umap_dim
		self.min_cluster_size = min_cluster_size
		self.fallback_similarity = fallback_similarity

	def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
		if not vectors:
			return []

		try:
			import numpy as np
			import umap  # type: ignore[import-not-found]
			from hdbscan import HDBSCAN  # type: ignore[import-not-found]

			x = np.asarray(vectors, dtype=float)
			reducer = umap.UMAP(
				n_components=self.umap_dim,
				metric="cosine",
				random_state=42,
				n_neighbors=min(30, max(5, len(vectors) // 10)),
				min_dist=0.0,
			)
			reduced = reducer.fit_transform(x)
			labels = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean").fit_predict(reduced)
			return [int(x) for x in labels.tolist()]
		except Exception:
			return SimilarityThresholdClusterer(self.fallback_similarity).fit_predict(vectors)


class SimilarityThresholdClusterer:
	"""无依赖回退聚类：基于中心相似度的贪心聚类。"""

	def __init__(self, threshold: float = 0.78):
		self.threshold = threshold

	def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
		if not vectors:
			return []

		labels = [-1] * len(vectors)
		centers: List[List[float]] = []
		counts: List[int] = []

		for i, vec in enumerate(vectors):
			best_idx = -1
			best_sim = -1.0
			for j, center in enumerate(centers):
				sim = cosine_similarity(vec, center)
				if sim > best_sim:
					best_sim = sim
					best_idx = j

			if best_idx >= 0 and best_sim >= self.threshold:
				labels[i] = best_idx
				counts[best_idx] += 1
				centers[best_idx] = running_mean_update(centers[best_idx], vec, counts[best_idx])
			else:
				labels[i] = len(centers)
				centers.append(list(vec))
				counts.append(1)

		return labels


class HeuristicNarrativeGenerator:
	"""无 LLM 时的内涵提取兜底实现。"""

	def generate(self, summaries: Sequence[str], domain: str) -> Dict[str, str]:
		text = "\n".join(summaries)
		top_terms = top_keywords(text, top_k=6)
		term = f"{domain}-{'/'.join(top_terms[:2])}" if top_terms else f"{domain}-技术簇"
		return {
			"technology_term": term,
			"technology_problem": "该技术簇聚焦提升性能、降低成本或解决现有流程瓶颈。",
			"technology_method": f"核心手段集中在：{', '.join(top_terms[:4]) if top_terms else '算法/工艺优化'}。",
			"application_direction": "预计应用于该领域的关键生产、研发或运营场景。",
		}


class LLMBasedNarrativeGenerator:
	"""基于外部 LLM 回调的内涵抽取。"""

	def __init__(self, llm_callable: Callable[[str], str]):
		self.llm_callable = llm_callable

	def generate(self, summaries: Sequence[str], domain: str) -> Dict[str, str]:
		prompt = (
			"你是一个资深科技情报分析师。请阅读以下从同一技术聚类中提取的专利/论文摘要集合，"
			"并严格输出 JSON，字段包括：technology_term, technology_problem, technology_method, application_direction。\n\n"
			f"领域：{domain}\n\n"
			"摘要集合：\n"
			+ "\n\n".join(summaries)
		)
		raw = self.llm_callable(prompt)
		data = safe_json_parse(raw)
		return {
			"technology_term": str(data.get("technology_term", "")).strip() or f"{domain}-候选技术",
			"technology_problem": str(data.get("technology_problem", "")).strip() or "未提供",
			"technology_method": str(data.get("technology_method", "")).strip() or "未提供",
			"application_direction": str(data.get("application_direction", "")).strip() or "未提供",
		}


class StaticExpertJudge:
	"""规则化专家（默认兜底）。"""

	def __init__(self, name: str):
		self.name = name

	def evaluate(self, cluster: ClusterInsight, domain: str) -> ExpertVote:
		del domain
		score = 0
		if cluster.trend.burst_detected:
			score += 1
		if cluster.trend.cagr > 0.15:
			score += 1
		if cluster.size >= 10:
			score += 1
		approve = score >= 2
		reason = "趋势和规模支持该方向" if approve else "趋势或规模不足，暂不建议入选"
		return ExpertVote(expert_name=self.name, approve=approve, reason=reason)


class LLMExpertJudge:
	"""可插拔 LLM 专家评审。"""

	def __init__(self, name: str, focus: str, llm_callable: Callable[[str], str]):
		self.name = name
		self.focus = focus
		self.llm_callable = llm_callable

	def evaluate(self, cluster: ClusterInsight, domain: str) -> ExpertVote:
		payload = {
			"domain": domain,
			"cluster": {
				"cluster_id": cluster.cluster_id,
				"size": cluster.size,
				"technology_term": cluster.technology_term,
				"technology_problem": cluster.technology_problem,
				"technology_method": cluster.technology_method,
				"application_direction": cluster.application_direction,
				"trend": {
					"yearly_count": cluster.trend.yearly_count,
					"yearly_share": cluster.trend.yearly_share,
					"cagr": cluster.trend.cagr,
					"burst_detected": cluster.trend.burst_detected,
				},
			},
			"focus": self.focus,
		}
		prompt = (
			"你是技术评审专家。请根据输入 JSON 给出是否通过（approve: true/false）和一句理由（reason）。"
			"只返回 JSON。\n"
			+ json.dumps(payload, ensure_ascii=False)
		)
		raw = self.llm_callable(prompt)
		data = safe_json_parse(raw)
		approve = bool(data.get("approve", False))
		reason = str(data.get("reason", "未提供理由")).strip()
		return ExpertVote(expert_name=self.name, approve=approve, reason=reason)


class TrendAnalyzer:
	"""按年统计 + CAGR + Burst Detection。"""

	def analyze(
		self,
		cluster_docs: Sequence[TimestampedDocument],
		all_domain_docs: Sequence[TimestampedDocument],
		burst_share_increase_threshold: float = 0.03,
	) -> TrendMetrics:
		cluster_yearly = count_by_year(cluster_docs)
		domain_yearly = count_by_year(all_domain_docs)

		years = sorted(set(cluster_yearly) | set(domain_yearly))
		yearly_share: Dict[int, float] = {}
		for y in years:
			cluster_cnt = cluster_yearly.get(y, 0)
			total_cnt = domain_yearly.get(y, 0)
			yearly_share[y] = (cluster_cnt / total_cnt) if total_cnt > 0 else 0.0

		cagr = compute_cagr(cluster_yearly)

		burst_detected = False
		burst_reason = "未检测到显著突现"
		if len(years) >= 3:
			y2, y1 = years[-2], years[-1]
			delta_share = yearly_share.get(y1, 0.0) - yearly_share.get(y2, 0.0)
			if delta_share >= burst_share_increase_threshold and cluster_yearly.get(y1, 0) >= cluster_yearly.get(y2, 0):
				burst_detected = True
				burst_reason = (
					f"近两年占比提升 {delta_share:.2%}，且文献量未下降，判定为技术突现"
				)

		return TrendMetrics(
			yearly_count=dict(sorted(cluster_yearly.items())),
			yearly_share=dict(sorted(yearly_share.items())),
			cagr=cagr,
			burst_detected=burst_detected,
			burst_reason=burst_reason,
		)


# =========================
# Main Agent
# =========================


class FrontierTechIdentificationAgent:
	"""jskxy-前沿技术识别智能体。"""

	def __init__(
		self,
		classifier: DomainClassifier,
		embedder: TextEmbedder,
		clusterer: Clusterer,
		narrative_generator: Optional[ClusterNarrativeGenerator] = None,
		trend_analyzer: Optional[TrendAnalyzer] = None,
		experts: Optional[Sequence[ExpertJudge]] = None,
		extractor: Optional[CoreTextExtractor] = None,
	):
		self.classifier = classifier
		self.embedder = embedder
		self.clusterer = clusterer
		self.narrative_generator = narrative_generator or HeuristicNarrativeGenerator()
		self.trend_analyzer = trend_analyzer or TrendAnalyzer()
		self.extractor = extractor or CoreTextExtractor()
		self.experts = list(experts) if experts else self._build_default_experts()

	def run(
		self,
		documents: Sequence[TimestampedDocument],
		domain: str,
		min_domain_probability: float = 0.6,
		top_k_per_cluster: int = 20,
		minimum_votes_to_pass: int = 3,
	) -> FrontierIdentificationReport:
		if not documents:
			return FrontierIdentificationReport(
				domain=domain,
				total_input_docs=0,
				kept_docs=0,
				removed_docs=0,
				clusters=[],
			)

		# 1) 文本瘦身
		core_texts = [self.extractor.build_core_text(doc) for doc in documents]

		# 2) 领域判别
		probs = self.classifier.predict_domain_probability(core_texts, domain)
		scored_docs = [
			DomainScoredDocument(document=doc, core_text=txt, domain_probability=p)
			for doc, txt, p in zip(documents, core_texts, probs)
		]

		# 3) 打标签 + 剔除杂音
		kept = [d for d in scored_docs if d.domain_probability >= min_domain_probability]
		removed = len(scored_docs) - len(kept)
		if not kept:
			return FrontierIdentificationReport(
				domain=domain,
				total_input_docs=len(documents),
				kept_docs=0,
				removed_docs=removed,
				clusters=[],
			)

		# 4) 向量化 + 降维聚类（在 clusterer 内部实现）
		kept_texts = [d.core_text for d in kept]
		vectors = self.embedder.embed(kept_texts)
		labels = self.clusterer.fit_predict(vectors)

		by_cluster: Dict[int, List[int]] = defaultdict(list)
		for idx, lb in enumerate(labels):
			by_cluster[int(lb)].append(idx)

		# noise label -1 直接剔除
		clusters: List[ClusterInsight] = []
		domain_docs = [d.document for d in kept]
		for cluster_id, idxs in sorted(by_cluster.items(), key=lambda kv: kv[0]):
			if cluster_id < 0:
				continue
			cluster_vectors = [vectors[i] for i in idxs]
			cluster_docs = [kept[i] for i in idxs]

			# 5) 提取每簇最核心 top-k 文献
			top_local_idx = top_k_nearest_to_centroid(cluster_vectors, top_k=min(top_k_per_cluster, len(cluster_docs)))
			representatives = [cluster_docs[i] for i in top_local_idx]
			rep_ids = [x.document.doc_id for x in representatives]
			rep_summaries = [x.document.abstract or x.core_text[:260] for x in representatives]

			# 6) 生成技术名词与技术内涵
			narrative = self.narrative_generator.generate(rep_summaries, domain)

			# 7) 时间演化特征
			trend = self.trend_analyzer.analyze(
				cluster_docs=[x.document for x in cluster_docs],
				all_domain_docs=domain_docs,
			)

			temp_cluster = ClusterInsight(
				cluster_id=cluster_id,
				size=len(cluster_docs),
				representative_doc_ids=rep_ids,
				representative_summaries=rep_summaries,
				technology_term=narrative["technology_term"],
				technology_problem=narrative["technology_problem"],
				technology_method=narrative["technology_method"],
				application_direction=narrative["application_direction"],
				trend=trend,
				votes=[],
				approved=False,
			)

			# 8) 五专家投票
			votes = [expert.evaluate(temp_cluster, domain) for expert in self.experts]
			approvals = sum(1 for v in votes if v.approve)
			temp_cluster.votes = votes
			temp_cluster.approved = approvals >= minimum_votes_to_pass

			if temp_cluster.approved:
				clusters.append(temp_cluster)

		return FrontierIdentificationReport(
			domain=domain,
			total_input_docs=len(documents),
			kept_docs=len(kept),
			removed_docs=removed,
			clusters=clusters,
		)

	@staticmethod
	def _build_default_experts() -> List[ExpertJudge]:
		return [
			StaticExpertJudge("专家A-学术界"),
			StaticExpertJudge("专家B-产业界"),
			StaticExpertJudge("专家C-专利局"),
			StaticExpertJudge("专家D-投资人"),
			StaticExpertJudge("专家E-政策专家"),
		]


class IdentificationAgent(ToolCallingAgent):
	"""可直接通过 `run()` 调用的前沿技术识别智能体。"""

	def __init__(
		self,
		model,
		tools: Optional[List] = None,
		classifier: Optional[DomainClassifier] = None,
		embedder: Optional[TextEmbedder] = None,
		clusterer: Optional[Clusterer] = None,
		narrative_generator: Optional[ClusterNarrativeGenerator] = None,
		trend_analyzer: Optional[TrendAnalyzer] = None,
		experts: Optional[Sequence[ExpertJudge]] = None,
		extractor: Optional[CoreTextExtractor] = None,
		**kwargs,
	):
		super().__init__(model=model, tools=tools or [], **kwargs)
		self.pipeline = FrontierTechIdentificationAgent(
			classifier=classifier or KeywordDomainClassifier(),
			embedder=embedder or HashingTextEmbedder(),
			clusterer=clusterer or UmapHdbscanClusterer(),
			narrative_generator=narrative_generator,
			trend_analyzer=trend_analyzer,
			experts=experts,
			extractor=extractor,
		)

	def _step_stream(self, memory_step: ActionStep):
		del memory_step
		try:
			self.logger.log_rule("🔎 前沿技术识别流程启动", level=LogLevel.INFO)
			payload = self._parse_task_payload(self.task)

			domain = str(payload.get("domain", "")).strip()
			raw_docs = payload.get("documents", [])
			if not domain:
				raise ValueError("task 中缺少 domain")
			if not isinstance(raw_docs, list) or not raw_docs:
				raise ValueError("task 中缺少 documents 列表")

			docs = [self._build_document(x) for x in raw_docs if isinstance(x, dict)]

			report = self.pipeline.run(
				documents=docs,
				domain=domain,
				min_domain_probability=float(payload.get("min_domain_probability", 0.6)),
				top_k_per_cluster=int(payload.get("top_k_per_cluster", 20)),
				minimum_votes_to_pass=int(payload.get("minimum_votes_to_pass", 3)),
			)

			report_dict = asdict(report)
			self.state["frontier_identification_report"] = report_dict
			yield ActionOutput(
				output=json.dumps(report_dict, ensure_ascii=False, indent=2),
				is_final_answer=True,
			)
		except Exception as exc:
			self.logger.log(f"❌ IdentificationAgent 执行失败: {exc}", level=LogLevel.ERROR)
			raise

	@staticmethod
	def _parse_task_payload(task: Any) -> Dict[str, Any]:
		if isinstance(task, dict):
			return task
		text = str(task or "").strip()
		if not text:
			return {}
		parsed = safe_json_parse(text)
		if parsed:
			return parsed
		# 允许传入 ```json ... ``` 包裹文本
		fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
		if fenced:
			return safe_json_parse(fenced.group(1))
		return {}

	@staticmethod
	def _build_document(item: Dict[str, Any]) -> TimestampedDocument:
		return TimestampedDocument(
			doc_id=str(item.get("doc_id", item.get("id", ""))).strip() or f"doc_{abs(hash(json.dumps(item, ensure_ascii=False))) % 10000000}",
			timestamp=str(item.get("timestamp", "")).strip(),
			text=str(item.get("text", "")).strip(),
			title=str(item.get("title", "")).strip(),
			abstract=str(item.get("abstract", "")).strip(),
			introduction_or_conclusion=str(item.get("introduction_or_conclusion", "")).strip(),
			independent_claim=str(item.get("independent_claim", "")).strip(),
			metadata=item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
		)


# =========================
# Utilities
# =========================


def safe_json_parse(text: str) -> Dict[str, Any]:
	raw = str(text or "").strip()
	if not raw:
		return {}
	try:
		data = json.loads(raw)
		return data if isinstance(data, dict) else {}
	except Exception:
		match = re.search(r"\{.*\}", raw, re.DOTALL)
		if not match:
			return {}
		try:
			data = json.loads(match.group(0))
			return data if isinstance(data, dict) else {}
		except Exception:
			return {}


def parse_year(ts: str) -> Optional[int]:
	if not ts:
		return None
	ts = ts.strip()
	m = re.search(r"(19|20)\d{2}", ts)
	if m:
		return int(m.group(0))
	try:
		return datetime.fromisoformat(ts).year
	except Exception:
		return None


def count_by_year(docs: Sequence[TimestampedDocument]) -> Dict[int, int]:
	c: Dict[int, int] = defaultdict(int)
	for d in docs:
		y = parse_year(d.timestamp)
		if y is not None:
			c[y] += 1
	return dict(c)


def compute_cagr(yearly_count: Dict[int, int]) -> float:
	if not yearly_count:
		return 0.0
	years = sorted(yearly_count)
	if len(years) < 2:
		return 0.0
	start_year, end_year = years[0], years[-1]
	periods = end_year - start_year
	if periods <= 0:
		return 0.0
	start_val = max(1, yearly_count.get(start_year, 0))
	end_val = max(0, yearly_count.get(end_year, 0))
	if end_val <= 0:
		return -1.0
	return (end_val / start_val) ** (1 / periods) - 1


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
	dot = sum(x * y for x, y in zip(a, b))
	na = math.sqrt(sum(x * x for x in a)) or 1.0
	nb = math.sqrt(sum(y * y for y in b)) or 1.0
	return dot / (na * nb)


def running_mean_update(current_mean: Sequence[float], new_vec: Sequence[float], count_after: int) -> List[float]:
	if count_after <= 1:
		return list(new_vec)
	old_count = count_after - 1
	return [((m * old_count) + x) / count_after for m, x in zip(current_mean, new_vec)]


def top_k_nearest_to_centroid(vectors: Sequence[Sequence[float]], top_k: int = 20) -> List[int]:
	if not vectors:
		return []
	dim = len(vectors[0])
	centroid = [0.0] * dim
	for v in vectors:
		for i, x in enumerate(v):
			centroid[i] += x
	n = len(vectors)
	centroid = [x / n for x in centroid]

	scored = [(i, cosine_similarity(v, centroid)) for i, v in enumerate(vectors)]
	scored.sort(key=lambda x: x[1], reverse=True)
	return [i for i, _ in scored[:top_k]]


def top_keywords(text: str, top_k: int = 6) -> List[str]:
	tokens = [
		t
		for t in re.split(r"[^\w\u4e00-\u9fff]+", text.lower())
		if t and len(t) >= 2 and not t.isdigit()
	]
	stopwords = {
		"the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
		"以及", "通过", "进行", "研究", "模型", "方法", "技术", "系统", "应用", "一种",
	}
	tokens = [t for t in tokens if t not in stopwords]
	if not tokens:
		return []
	cnt = Counter(tokens)
	return [w for w, _ in cnt.most_common(top_k)]


__all__ = [
	"TimestampedDocument",
	"DomainScoredDocument",
	"TrendMetrics",
	"ExpertVote",
	"ClusterInsight",
	"FrontierIdentificationReport",
	"DomainClassifier",
	"TextEmbedder",
	"Clusterer",
	"ClusterNarrativeGenerator",
	"ExpertJudge",
	"CoreTextExtractor",
	"KeywordDomainClassifier",
	"SklearnRandomForestDomainClassifier",
	"HashingTextEmbedder",
	"BGEM3Embedder",
	"UmapHdbscanClusterer",
	"SimilarityThresholdClusterer",
	"HeuristicNarrativeGenerator",
	"LLMBasedNarrativeGenerator",
	"StaticExpertJudge",
	"LLMExpertJudge",
	"TrendAnalyzer",
	"FrontierTechIdentificationAgent",
	"IdentificationAgent",
	"top_k_nearest_to_centroid",
	"compute_cagr",
]

