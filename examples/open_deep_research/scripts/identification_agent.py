# from __future__ import annotations

# import ast
# import json
# import math
# import re
# from collections import Counter, defaultdict
# from dataclasses import asdict, dataclass, field
# from datetime import datetime
# import time
# from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

# import concurrent

# from smolagents import ToolCallingAgent
# from smolagents.agents import ActionOutput, CustomAgent, ToolOutput
# from smolagents.memory import ActionStep
# from smolagents.monitoring import LogLevel
# from utils.common_utils import safe_json_parse


# # =========================
# # Data Models
# # =========================
# import os
# import logging

# def get_file_logger(log_path: str = "outputs/identification_agent.log") -> logging.Logger:
#     """配置文件日志记录器（全量细节）"""
#     # 确保 outputs 目录存在
#     os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
#     logger = logging.getLogger("AgentFileLogger")
#     logger.setLevel(logging.INFO)
    
#     # 避免重复添加 handler
#     if not logger.handlers:
#         fh = logging.FileHandler(log_path, encoding='utf-8')
#         formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
        
#     return logger

# @dataclass
# class TimestampedDocument:
#     """输入文档对象。"""

#     doc_id: str
#     timestamp: str
#     text: str = ""
#     title: str = ""
#     abstract: str = ""
#     introduction_or_conclusion: str = ""
#     independent_claim: str = ""
#     metadata: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class DomainScoredDocument:
#     """带领域置信度的文档。"""

#     document: TimestampedDocument
#     core_text: str
#     domain_probability: float


# @dataclass
# class TrendMetrics:
#     """技术簇时间演化指标。"""

#     yearly_count: Dict[int, int]
#     yearly_share: Dict[int, float]
#     cagr: float
#     burst_detected: bool
#     burst_reason: str


# @dataclass
# class ExpertVote:
#     """单个专家评审结果。"""

#     expert_name: str
#     approve: bool
#     reason: str


# @dataclass
# class ClusterInsight:
#     """单个聚类簇洞察。"""

#     cluster_id: int
#     size: int
#     representative_doc_ids: List[str]
#     representative_summaries: List[str]
#     technology_term: str
#     technology_problem: str
#     technology_method: str
#     application_direction: str
#     trend: TrendMetrics
#     votes: List[ExpertVote]
#     approved: bool


# @dataclass
# class FrontierIdentificationReport:
#     """最终输出报告对象。"""

#     domain: str
#     total_input_docs: int
#     kept_docs: int
#     removed_docs: int
#     clusters: List[ClusterInsight]


# # =========================
# # Protocols / Interfaces
# # =========================


# class DomainClassifier(Protocol):
#     """领域判别器接口。"""

#     def predict_domain_probability(self, texts: Sequence[str], domain: str) -> List[float]:
#         ...


# class TextEmbedder(Protocol):
#     """向量化接口。"""

#     def embed(self, texts: Sequence[str]) -> List[List[float]]:
#         ...


# class Clusterer(Protocol):
#     """聚类接口。"""

#     def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
#         ...


# class ClusterNarrativeGenerator(Protocol):
#     """技术簇内涵抽取接口。"""

#     def generate(self, summaries: Sequence[str], domain: str) -> Dict[str, str]:
#         ...


# class ExpertJudge(Protocol):
#     """专家评审接口。"""

#     def evaluate(self, cluster: ClusterInsight, domain: str) -> ExpertVote:
#         ...


# # =========================
# # Core Components
# # =========================


# class CoreTextExtractor:
#     """文本瘦身器：标题 + 摘要 + 权利要求/引言结论。"""

#     def build_core_text(self, doc: TimestampedDocument, max_chars: int = 1400) -> str:
#         parts: List[str] = []
#         if doc.title:
#             parts.append(f"[标题] {doc.title.strip()}")
#         if doc.abstract:
#             parts.append(f"[摘要] {doc.abstract.strip()}")
#         if doc.independent_claim:
#             parts.append(f"[独立权利要求] {doc.independent_claim.strip()}")
#         if doc.introduction_or_conclusion:
#             parts.append(f"[引言/结论] {doc.introduction_or_conclusion.strip()}")

#         if not parts and doc.text:
#             parts.append(doc.text.strip())

#         merged = "\n".join(parts).strip()
#         if len(merged) <= max_chars:
#             return merged
#         return merged[:max_chars] + "..."


# class SmallModelDomainClassifier:
#     """基于小参数模型的并发批量领域判别器（带重试机制）"""

#     def __init__(
#         self, 
#         llm_callable: Callable[[str], str], 
#         batch_size: int = 5, 
#         max_workers: int = 10,
#         log_callable=None
#     ):
#         self.llm_callable = llm_callable
#         self.batch_size = batch_size
#         self.max_workers = max_workers
#         self.log = log_callable or (lambda m, l: None)

#     def _judge_batch(self, batch_texts: List[str], domain: str) -> List[float]:
#         texts_str = "\n".join([f"[{i}] {text[:600]}..." for i, text in enumerate(batch_texts)])
        
#         base_prompt = (
#             f"你是一个严格的学术文献领域分类器。请判断以下 {len(batch_texts)} 篇文献是否属于【{domain}】领域。\n"
#             f"请仔细阅读每篇文献的核心内容，并给出一个 0.0 到 1.0 之间的置信度分数（0.0表示完全无关，1.0表示高度相关）。\n"
#             f"【要求】严格返回一个纯 JSON 数组，数组长度必须刚好为 {len(batch_texts)}，每个元素是一个浮点数。\n"
#             f"示例输出格式：[0.95, 0.12, 0.88]\n\n"
#             f"文献列表：\n{texts_str}"
#         )

#         prompt = base_prompt
#         for attempt in range(3):
#             raw_response = self.llm_callable(prompt)
            
#             # ✨ 一行代码搞定解析
#             scores = safe_json_parse(raw_response, fallback_type=list)
            
#             if scores and len(scores) == len(batch_texts):
#                 try:
#                     return [float(s) for s in scores]
#                 except ValueError:
#                     pass # 捕获列表中包含非数字的极端情况
                    
#             self.log(f"⚠️ 模型 JSON 解析失败或长度不对 (尝试 {attempt+1}/3)", LogLevel.INFO)
#             prompt = base_prompt + "\n\n【系统警告】上一次输出格式错误。请务必只输出长度为 {} 的纯数字 JSON 数组！".format(len(batch_texts))

#         self.log(f"❌ 模型 3 次重试均失败，该批次使用默认保底分。", LogLevel.ERROR)
#         return [0.5] * len(batch_texts)

#     def predict_domain_probability(self, texts: Sequence[str], domain: str) -> List[float]:
#         if not texts:
#             return []

#         self.log(f"🧠 启动模型领域判别 (共 {len(texts)} 篇, 批大小: {self.batch_size}, 并发: {self.max_workers})", LogLevel.INFO)

#         batches = [list(texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
#         all_scores = []
        
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [executor.submit(self._judge_batch, batch, domain) for batch in batches]
#             for future in futures:
#                 all_scores.extend(future.result())

#         return all_scores[:len(texts)]


# class BGEM3Embedder:
#     """BGE-M3 向量器（sentence-transformers + GPU）。"""

#     def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 128):
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self._model = None

#     def _ensure_model(self) -> None:
#         if self._model is not None:
#             return
#         try:
#             from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
#         except Exception as exc:  # pragma: no cover
#             raise ImportError("需要安装 sentence-transformers 才能使用 BGEM3Embedder") from exc
#         self._model = SentenceTransformer(self.model_name)

#     def embed(self, texts: Sequence[str]) -> List[List[float]]:
#         self._ensure_model()
#         arr = self._model.encode(
#             list(texts),
#             batch_size=self.batch_size,
#             show_progress_bar=False,
#             normalize_embeddings=True,
#         )
#         return [list(map(float, row)) for row in arr]


# class UmapHdbscanClusterer:
#     """UMAP 降维 + HDBSCAN 聚类"""

#     def __init__(self, umap_dim: int = 10, min_cluster_size: int = 8, log_callable=None):
#         self.umap_dim = umap_dim
#         self.min_cluster_size = min_cluster_size
#         self.log = log_callable or (lambda m, l: None)

#     def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
#         if not vectors:
#             return []

#         import numpy as np
#         import umap  # type: ignore[import-not-found]
#         from hdbscan import HDBSCAN  # type: ignore[import-not-found]

#         x = np.asarray(vectors, dtype=float)
#         reducer = umap.UMAP(
#             n_components=self.umap_dim,
#             metric="cosine",
#             random_state=42,
#             n_jobs=1,
#             n_neighbors=min(30, max(5, len(vectors) // 10)),
#             min_dist=0.0,
#         )
#         reduced = reducer.fit_transform(x)
#         labels = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean").fit_predict(reduced)
#         return [int(x) for x in labels.tolist()]


# class ModelNarrativeGenerator:
#     """通过模型API进行真正的技术内涵提取（带重试机制）"""

#     def __init__(self, llm_callable: Callable[[str], str], log_callable=None):
#         self.llm_callable = llm_callable
#         self.log = log_callable or (lambda m, l: None)

#     def generate(self, summaries: Sequence[str], domain: str) -> Dict[str, str]:
#         base_prompt = (
#             "你是一个资深科技情报分析师。请阅读以下从同一技术聚类中提取的专利/论文摘要集合，"
#             "并严格输出 JSON 格式，提取该技术簇的核心信息。包含以下字段：\n"
#             "1. technology_term: 前沿技术名词（要求：必须具体、具象且有高辨识度，严禁使用如'Agentic Workflow'这种过于泛化的通用词汇）\n"
#             "2. technology_problem: 技术试图解决的具体痛点\n"
#             "3. technology_method: 采用的新型材料、算法或工艺架构\n"
#             "4. application_direction: 主要应用的商业或工业场景\n\n"
#             f"目标领域：{domain}\n\n"
#             "摘要集合：\n"
#             + "\n\n".join(summaries)
#         )
        
#         prompt = base_prompt
#         # 最大重试 3 次
#         for attempt in range(3):
#             raw = self.llm_callable(prompt)
#             try:
#                 clean_str = raw.replace("```json", "").replace("```", "").strip()
#                 data = json.loads(clean_str)
                
#                 # 校验核心字段
#                 if "technology_term" in data:
#                     term = str(data.get("technology_term", "")).strip() or f"{domain}-候选技术"
#                     self.log(f"模型提炼出的技术名词为: {term}", LogLevel.INFO)
#                     return {
#                         "technology_term": term,
#                         "technology_problem": str(data.get("technology_problem", "")).strip() or "未提供",
#                         "technology_method": str(data.get("technology_method", "")).strip() or "未提供",
#                         "application_direction": str(data.get("application_direction", "")).strip() or "未提供",
#                     }
#                 else:
#                     raise ValueError("JSON 中缺少 technology_term 字段")
                    
#             except Exception as e:
#                 self.log(f"⚠️ 内涵提取 JSON 解析失败 (尝试 {attempt+1}/3): {e}", LogLevel.INFO)
#                 prompt = base_prompt + f"\n\n【系统警告】你上一次的输出解析失败（错误：{e}）。请严格返回包含要求的 4 个字段的纯 JSON 格式！"
        
#         # 3 次全失败，返回降级数据
#         self.log("❌ 内涵提取 3 次重试均失败，返回降级默认值。", LogLevel.ERROR)
#         return {
#             "technology_term": f"{domain}-未命名候选技术",
#             "technology_problem": "大模型解析失败",
#             "technology_method": "大模型解析失败",
#             "application_direction": "大模型解析失败",
#         }


# class ModelExpertJudge:
#     """通过模型API进行独立投票打分的评审专家（带重试机制）"""

#     def __init__(self, name: str, focus: str, llm_callable: Callable[[str], str], log_callable=None):
#         self.name = name
#         self.focus = focus
#         self.llm_callable = llm_callable
#         self.log = log_callable or (lambda m, l: None)

#     def evaluate(self, cluster: ClusterInsight, domain: str) -> ExpertVote:
#         payload = {
#             "domain": domain,
#             "cluster": {
#                 "cluster_id": cluster.cluster_id,
#                 "size": cluster.size,
#                 "technology_term": cluster.technology_term,
#                 "technology_problem": cluster.technology_problem,
#                 "technology_method": cluster.technology_method,
#                 "application_direction": cluster.application_direction,
#                 "trend": {
#                     "yearly_count": cluster.trend.yearly_count,
#                     "yearly_share": cluster.trend.yearly_share,
#                     "cagr": cluster.trend.cagr,
#                     "burst_detected": cluster.trend.burst_detected,
#                 },
#             },
#             "expert_focus": self.focus,
#         }
        
#         base_prompt = (
#             f"你是【{self.name}】。你的评审侧重点是：{self.focus}。\n"
#             "请根据输入的数据簇 JSON 给出是否同意其作为前沿技术输出到最终报告中。\n"
#             "请严格以纯 JSON 格式返回结果，必须包含以下两个字段：\n"
#             "1. approve: boolean (true 或 false)\n"
#             "2. reason: string (一句话精炼评价理由)\n\n"
#             "输入数据簇：\n"
#             + json.dumps(payload, ensure_ascii=False)
#         )
        
#         prompt = base_prompt
#         # 最大重试 3 次
#         for attempt in range(3):
#             raw = self.llm_callable(prompt)
#             try:
#                 clean_str = raw.replace("```json", "").replace("```", "").strip()
#                 data = json.loads(clean_str)
                
#                 # 校验关键字段是否存在
#                 if "approve" in data and "reason" in data:
#                     approve = bool(data.get("approve", False))
#                     reason = str(data.get("reason", "")).strip()
#                     self.log(f"专家 [{self.name}] 评审完成 - 投票: {'赞成' if approve else '反对'} - 理由: {reason}", LogLevel.INFO)
#                     return ExpertVote(expert_name=self.name, approve=approve, reason=reason)
#                 else:
#                     raise ValueError("JSON 中缺少 approve 或 reason 字段")
                    
#             except Exception as e:
#                 self.log(f"⚠️ 专家 [{self.name}] JSON 解析失败 (尝试 {attempt+1}/3): {e}", LogLevel.INFO)
#                 prompt = base_prompt + f"\n\n【系统警告】你上一次的输出解析失败（错误：{e}）。请务必只输出包含 approve 和 reason 的合法 JSON，禁止输出任何多余的分析文字！"
                
#         # 3 次全失败，默认一票否决
#         self.log(f"❌ 专家 [{self.name}] 3 次重试均失败，默认投反对票。", LogLevel.ERROR)
#         return ExpertVote(expert_name=self.name, approve=False, reason="模型格式持续输出错误，系统默认否决")


# class TrendAnalyzer:
#     """按年统计 + CAGR + Burst Detection。"""

#     def analyze(
#         self,
#         cluster_docs: Sequence[TimestampedDocument],
#         all_domain_docs: Sequence[TimestampedDocument],
#         burst_share_increase_threshold: float = 0.03,
#     ) -> TrendMetrics:
#         cluster_yearly = count_by_year(cluster_docs)
#         domain_yearly = count_by_year(all_domain_docs)

#         years = sorted(set(cluster_yearly) | set(domain_yearly))
#         yearly_share: Dict[int, float] = {}
#         for y in years:
#             cluster_cnt = cluster_yearly.get(y, 0)
#             total_cnt = domain_yearly.get(y, 0)
#             yearly_share[y] = (cluster_cnt / total_cnt) if total_cnt > 0 else 0.0

#         cagr = compute_cagr(cluster_yearly)

#         burst_detected = False
#         burst_reason = "未检测到显著突现"
#         if len(years) >= 3:
#             y2, y1 = years[-2], years[-1]
#             delta_share = yearly_share.get(y1, 0.0) - yearly_share.get(y2, 0.0)
#             if delta_share >= burst_share_increase_threshold and cluster_yearly.get(y1, 0) >= cluster_yearly.get(y2, 0):
#                 burst_detected = True
#                 burst_reason = (
#                     f"近两年占比提升 {delta_share:.2%}，且文献量未下降，判定为技术突现"
#                 )

#         return TrendMetrics(
#             yearly_count=dict(sorted(cluster_yearly.items())),
#             yearly_share=dict(sorted(yearly_share.items())),
#             cagr=cagr,
#             burst_detected=burst_detected,
#             burst_reason=burst_reason,
#         )


# # =========================
# # Main Agent
# # =========================


# class FrontierTechIdentificationAgent:
#     """前沿技术识别核心管道。"""

#     def __init__(
#         self,
#         classifier: DomainClassifier,
#         embedder: TextEmbedder,
#         clusterer: Clusterer,
#         narrative_generator: ClusterNarrativeGenerator,
#         experts: Sequence[ExpertJudge],
#         trend_analyzer: Optional[TrendAnalyzer] = None,
#         extractor: Optional[CoreTextExtractor] = None,
#         log_callable: Optional[Callable[[str, Any], None]] = None
#     ):
#         self.classifier = classifier
#         self.embedder = embedder
#         self.clusterer = clusterer
#         self.narrative_generator = narrative_generator
#         self.trend_analyzer = trend_analyzer or TrendAnalyzer()
#         self.extractor = extractor or CoreTextExtractor()
#         self.experts = list(experts)
#         self.log = log_callable or (lambda m, l: None)

#     def run(
#         self,
#         documents: Sequence[TimestampedDocument],
#         domain: str,
#         min_domain_probability: float = 0.6,
#         top_k_per_cluster: int = 20,
#         minimum_votes_to_pass: int = 3,
#     ) -> FrontierIdentificationReport:
        
#         self.log(f"🚀 开始执行前沿技术识别 | 目标领域: {domain} | 输入文档总数: {len(documents)}", LogLevel.INFO)
        
#         if not documents:
#             return FrontierIdentificationReport(
#                 domain=domain,
#                 total_input_docs=0,
#                 kept_docs=0,
#                 removed_docs=0,
#                 clusters=[],
#             )

#         # 1) 文本瘦身
#         self.log(f"📥 [CoreTextExtractor 输入] 文档数量: {len(documents)}", LogLevel.INFO)
#         core_texts = [self.extractor.build_core_text(doc) for doc in documents]
#         self.log(f"📤 [CoreTextExtractor 输出] 瘦身后首个文档长度: {len(core_texts[0]) if core_texts else 0} 字符", LogLevel.INFO)

#         # 2) 领域判别
#         self.log(f"📥 [DomainClassifier 输入] 待判别文本数量: {len(core_texts)}", LogLevel.INFO)
#         probs = self.classifier.predict_domain_probability(core_texts, domain)
#         self.log(f"📤 [DomainClassifier 输出] 判别分数列表(前5个): {probs[:5]}", LogLevel.INFO)

#         # 2) 领域判别
#         probs = self.classifier.predict_domain_probability(core_texts, domain)
#         scored_docs = [
#             DomainScoredDocument(document=doc, core_text=txt, domain_probability=p)
#             for doc, txt, p in zip(documents, core_texts, probs)
#         ]

#         # 3) 打标签 + 剔除杂音
#         kept = [d for d in scored_docs if d.domain_probability >= min_domain_probability]
#         removed = len(scored_docs) - len(kept)
#         self.log(f"领域判别完成 -> 保留: {len(kept)} 篇 | 剔除: {removed} 篇", LogLevel.INFO)
        
#         if not kept:
#             return FrontierIdentificationReport(
#                 domain=domain,
#                 total_input_docs=len(documents),
#                 kept_docs=0,
#                 removed_docs=removed,
#                 clusters=[],
#             )

#         # 4) 向量化 + 降维聚类
#         # 4) 向量化 + 降维聚类
#         kept_texts = [d.core_text for d in kept]
#         self.log(f"📥 [TextEmbedder 输入] 待向量化文本数: {len(kept_texts)}", LogLevel.INFO)
#         vectors = self.embedder.embed(kept_texts)
#         self.log(f"📤 [TextEmbedder 输出] 向量矩阵形状: {len(vectors)} x {len(vectors[0]) if vectors else 0}", LogLevel.INFO)

#         self.log(f"📥 [Clusterer 输入] 向量矩阵行数: {len(vectors)}", LogLevel.INFO)
#         labels = self.clusterer.fit_predict(vectors)
#         self.log(f"📤 [Clusterer 输出] 聚类标签分布: {dict(Counter(labels))}", LogLevel.INFO)

#         by_cluster: Dict[int, List[int]] = defaultdict(list)
#         for idx, lb in enumerate(labels):
#             by_cluster[int(lb)].append(idx)

#         valid_clusters_count = len([c for c in by_cluster.keys() if c >= 0])
#         noise_docs_count = len(by_cluster.get(-1, []))
#         self.log(f"聚类完成 -> 发现 {valid_clusters_count} 个有效技术簇 | 判定为噪声的文档: {noise_docs_count} 篇", LogLevel.INFO)

#         # noise label -1 直接剔除
#         clusters: List[ClusterInsight] = []
#         domain_docs = [d.document for d in kept]
#         for cluster_id, idxs in sorted(by_cluster.items(), key=lambda kv: kv[0]):
#             if cluster_id < 0:
#                 continue
            
#             self.log(f"  ▶ 正在处理簇 [{cluster_id}] (包含 {len(idxs)} 篇文档)", LogLevel.INFO)
#             cluster_vectors = [vectors[i] for i in idxs]
#             cluster_docs = [kept[i] for i in idxs]

#             # 5) 提取每簇最核心 top-k 文献
#             top_local_idx = top_k_nearest_to_centroid(cluster_vectors, top_k=min(top_k_per_cluster, len(cluster_docs)))
#             representatives = [cluster_docs[i] for i in top_local_idx]
#             rep_ids = [x.document.doc_id for x in representatives]
#             rep_summaries = [x.document.abstract or x.core_text[:260] for x in representatives]

#             # 6) 生成技术名词与技术内涵
#             narrative = self.narrative_generator.generate(rep_summaries, domain)

#             # 7) 时间演化特征
#             self.log(f"📥 [TrendAnalyzer 输入] 簇文档数: {len(cluster_docs)}, 全局文档数: {len(domain_docs)}", LogLevel.INFO)
#             trend = self.trend_analyzer.analyze(
#                 cluster_docs=[x.document for x in cluster_docs],
#                 all_domain_docs=domain_docs,
#             )
#             self.log(f"📤 [TrendAnalyzer 输出] CAGR: {trend.cagr:.2f}, 突现: {trend.burst_detected} ({trend.burst_reason})", LogLevel.INFO)

#             temp_cluster = ClusterInsight(
#                 cluster_id=cluster_id,
#                 size=len(cluster_docs),
#                 representative_doc_ids=rep_ids,
#                 representative_summaries=rep_summaries,
#                 technology_term=narrative["technology_term"],
#                 technology_problem=narrative["technology_problem"],
#                 technology_method=narrative["technology_method"],
#                 application_direction=narrative["application_direction"],
#                 trend=trend,
#                 votes=[],
#                 approved=False,
#             )

#             # 8) 五专家并发投票
#             self.log(f"    - 正在并发请求 {len(self.experts)} 位专家进行评审...", LogLevel.INFO)
            
#             with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.experts)) as executor:
#                 # 提交所有专家的评审任务
#                 futures = [executor.submit(expert.evaluate, temp_cluster, domain) for expert in self.experts]
#                 # 阻塞等待所有结果返回（按提交顺序收集结果）
#                 votes = [future.result() for future in futures]
                
#             approvals = sum(1 for v in votes if v.approve)
#             temp_cluster.votes = votes
#             temp_cluster.approved = approvals >= minimum_votes_to_pass
            
#             status_str = "通过 ✅" if temp_cluster.approved else "未通过 ❌"
#             self.log(f"    - 评审结果: {approvals}/{len(self.experts)} 票赞成 -> {status_str}", LogLevel.INFO)
#             if temp_cluster.approved:
#                 clusters.append(temp_cluster)

#         self.log(f"🏁 流程结束 | 最终共选出 {len(clusters)} 个获批的前沿技术。", LogLevel.INFO)
#         return FrontierIdentificationReport(
#             domain=domain,
#             total_input_docs=len(documents),
#             kept_docs=len(kept),
#             removed_docs=removed,
#             clusters=clusters,
#         )


# class IdentificationAgent(CustomAgent):
#     """可直接通过 `run()` 调用的前沿技术识别智能体。"""

#     def __init__(
#         self,
#         model,
#         small_model=None,
#         tools: Optional[List] = None,
#         classifier: Optional[DomainClassifier] = None,
#         embedder: Optional[TextEmbedder] = None,
#         clusterer: Optional[Clusterer] = None,
#         narrative_generator: Optional[ClusterNarrativeGenerator] = None,
#         trend_analyzer: Optional[TrendAnalyzer] = None,
#         experts: Optional[Sequence[ExpertJudge]] = None,
#         extractor: Optional[CoreTextExtractor] = None,
#         **kwargs,
#     ):
#         super().__init__(model=model, tools=tools or [], **kwargs)
#         self.run_dir = f"outputs/identification_{time.strftime('%Y%m%d_%H%M%S')}"
#         os.makedirs(self.run_dir, exist_ok=True)
#         # 1. 挂载文件日志
#         file_logger = get_file_logger(os.path.join(self.run_dir, "agent_run.log"))

#         # 2. 封装双写日志闭包
#         def agent_logger(msg: str, level: LogLevel = LogLevel.INFO):
#             # 记录完整信息到文件
#             py_level = logging.ERROR if level == LogLevel.ERROR else logging.INFO
#             file_logger.log(py_level, msg)

#             # 控制台输出精简逻辑
#             console_msg = msg
#             max_console_len = 200  # 控制台最大显示字符数阈值
            
#             # 如果文本过长（尤其是大模型 prompt 和 json 返回），进行截断
#             if len(msg) > max_console_len:
#                 # 保留头尾或者只保留头部，这里选择保留前 200 个字符并提示
#                 console_msg = msg[:max_console_len].strip() + f"\n... [已截断，省略 {len(msg) - max_console_len} 字符，详情见 agent_run.log]"
            
#             # 调用 smolagents 自身的 logger 打印到控制台
#             self.logger.log(console_msg, level=level)
        
#         # 封装日志闭包向下传递
#         def agent_logger(msg: str, level: LogLevel = LogLevel.INFO):
#             self.logger.log(msg, level=level)

#         def invoke_llm(prompt: str) -> str:
#             messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
#             agent_logger(f"\n========== 🟢 [大模型调用输入] ==========\n{prompt}\n==========================================", LogLevel.INFO)
#             try:
#                 response = self.model(messages)
#                 res_text = response.content if hasattr(response, "content") else str(response)
#                 agent_logger(f"\n========== 🔴 [大模型调用输出] ==========\n{res_text}\n==========================================", LogLevel.INFO)
#                 return res_text
#             except Exception as e:
#                 agent_logger(f"Model API 调用异常: {e}", LogLevel.ERROR)
#                 return "{}"
        
#         def invoke_small_llm(prompt: str) -> str:
#             target_model = small_model if small_model else self.model
#             messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
#             agent_logger(f"\n========== 🟢 [小模型调用输入] ==========\n{prompt}\n==========================================", LogLevel.INFO)
#             try:
#                 response = target_model(messages)
#                 res_text = response.content if hasattr(response, "content") else str(response)
#                 agent_logger(f"\n========== 🔴 [小模型调用输出] ==========\n{res_text}\n==========================================", LogLevel.INFO)
#                 return res_text
#             except Exception as e:
#                 agent_logger(f"小模型 API 调用异常: {e}", LogLevel.ERROR)
#                 return "{}"

#         default_narrative_generator = narrative_generator or ModelNarrativeGenerator(invoke_llm, log_callable=agent_logger)

#         default_experts = experts or [
#             ModelExpertJudge("专家A-学术界", "技术的理论创新性和突破性", invoke_llm, log_callable=agent_logger),
#             ModelExpertJudge("专家B-产业界", "技术的落地可行性和商业转化潜力", invoke_llm, log_callable=agent_logger),
#             ModelExpertJudge("专家C-专利局", "该技术是否具有足够的独创性，且未被广泛公知", invoke_llm, log_callable=agent_logger),
#             ModelExpertJudge("专家D-投资人", "该技术的市场规模潜力和爆发时间点", invoke_llm, log_callable=agent_logger),
#             ModelExpertJudge("专家E-政策专家", "该技术是否符合宏观产业政策导向", invoke_llm, log_callable=agent_logger),
#         ]

#         # 实例化模型判别器
#         default_classifier = SmallModelDomainClassifier(
#             llm_callable=invoke_llm, 
#             batch_size=5, 
#             max_workers=10, 
#             log_callable=agent_logger
#         )

#         self.pipeline = FrontierTechIdentificationAgent(
#             classifier=classifier or default_classifier,
#             embedder=embedder or BGEM3Embedder(batch_size=64),
#             clusterer=clusterer or UmapHdbscanClusterer(log_callable=agent_logger,min_cluster_size=3),
#             narrative_generator=default_narrative_generator,
#             experts=default_experts,
#             trend_analyzer=trend_analyzer,
#             extractor=extractor,
#             log_callable=agent_logger
#         )

#     def _step_stream(self, memory_step: ActionStep):
#         del memory_step
#         try:
#             self.logger.log_rule("🔎 前沿技术识别流程启动", level=LogLevel.INFO)
#             payload = self._parse_task_payload(self.task)

#             domain = str(payload.get("domain", "")).strip()
#             raw_docs = payload.get("documents", [])
#             if not domain:
#                 raise ValueError("task 中缺少 domain")
#             if not isinstance(raw_docs, list) or not raw_docs:
#                 raise ValueError("task 中缺少 documents 列表")

#             docs = [self._build_document(x) for x in raw_docs if isinstance(x, dict)]

#             # 1. 执行核心的识别管道，产出结构化报告对象
#             report = self.pipeline.run(
#                 documents=docs,
#                 domain=domain,
#                 min_domain_probability=float(payload.get("min_domain_probability", 0.6)),
#                 top_k_per_cluster=int(payload.get("top_k_per_cluster", 20)),
#                 minimum_votes_to_pass=int(payload.get("minimum_votes_to_pass", 3)),
#             )

#             report_dict = asdict(report)
#             self.state["frontier_identification_report"] = report_dict
#             raw_json = json.dumps(report_dict, ensure_ascii=False, indent=2)

#             # 2. 管道执行完后，主动调用一次大模型，将 JSON 转化为可读研报
#             self.logger.log("📝 正在调用大模型将识别结果结构化为可读研报...", level=LogLevel.INFO)
            
#             final_prompt = (
#                 f"请根据以下前沿技术识别的 JSON 结果，撰写一份通俗易懂的分析报告。\n\n"
#                 f"【JSON 数据】\n{raw_json}\n\n"
#                 f"【输出要求】\n"
#                 f"1. 请为你生成的报告拟定一个简洁、有力的主标题，**严禁使用任何带有冒号（: 或 ：）的格式**。\n"
#                 f"2. 在技术簇的描述中，务必将JSON数据中的**专家投票数（如 X/5 票）**以及**CAGR（复合年增长率）**等量化数据直接融入正文说明中，以增强报告的说服力。\n"
#                 f"3. 报告应当结构清晰，语言通俗易懂，适合非技术背景的决策者阅读。\n"
#             )

#             messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
#             self.logger.log(f"\n========== 🟢 [最终研报生成输入] ==========\n{final_prompt}\n==========================================", LogLevel.INFO)
#             try:
#                 response = self.model(messages)
#                 final_report = response.content if hasattr(response, "content") else str(response)
#                 self.logger.log(f"\n========== 🔴 [最终研报生成输出] ==========\n{final_report}\n==========================================", LogLevel.INFO)
#             except Exception as e:
#                 self.logger.log(f"生成最终报告时调用大模型失败: {e}", level=LogLevel.ERROR)
#                 final_report = f"报告生成失败，原始 JSON 数据如下：\n{raw_json}"

#             # 3. 把最终的大模型生成的文本作为 final_answer 返回
#             self.logger.log_rule("✅ 报告生成完毕", level=LogLevel.INFO)
#             with open(os.path.join(self.run_dir, "frontier_identification_report.json"), "w", encoding="utf-8") as f:
#                 f.write(raw_json)
#             with open(os.path.join(self.run_dir, "frontier_identification_report.md"), "w", encoding="utf-8") as f:
#                 f.write(final_report)
#             self.logger.log(f"💾 识别结果及最终报告已保存至 {self.run_dir}/ 文件夹", level=LogLevel.INFO)
#             yield ActionOutput(
#                 output=final_report,
#                 is_final_answer=True,
#             )

#         except Exception as exc:
#             self.logger.log(f"❌ IdentificationAgent 执行失败: {exc}", level=LogLevel.ERROR)
#             raise

#     @staticmethod
#     def _parse_task_payload(task: Any) -> Dict[str, Any]:
#         if isinstance(task, dict):
#             return task

#         text = str(task or "").strip()
#         if not text:
#             return {}

#         stack = []
#         start = None
#         candidate = None

#         for i, ch in enumerate(text):
#             if ch == "{":
#                 if not stack:
#                     start = i
#                 stack.append(ch)
#             elif ch == "}":
#                 if stack:
#                     stack.pop()
#                     if not stack and start is not None:
#                         candidate = text[start:i + 1]

#         if not candidate:
#             return {}

#         parsed = safe_json_parse(candidate)
#         if parsed:
#             return parsed

#         try:
#             data = ast.literal_eval(candidate)
#             return data if isinstance(data, dict) else {}
#         except Exception as e:
#             print("ast parse failed:", e)

#         return {}

#     @staticmethod
#     def _build_document(item: Dict[str, Any]) -> TimestampedDocument:
#         return TimestampedDocument(
#             doc_id=str(item.get("doc_id", item.get("id", ""))).strip() or f"doc_{abs(hash(json.dumps(item, ensure_ascii=False))) % 10000000}",
#             timestamp=str(item.get("timestamp", "")).strip(),
#             text=str(item.get("text", "")).strip(),
#             title=str(item.get("title", "")).strip(),
#             abstract=str(item.get("abstract", "")).strip(),
#             introduction_or_conclusion=str(item.get("introduction_or_conclusion", "")).strip(),
#             independent_claim=str(item.get("independent_claim", "")).strip(),
#             metadata=item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
#         )


# # =========================
# # Utilities
# # =========================


# def parse_year(ts: str) -> Optional[int]:
#     if not ts:
#         return None
#     ts = ts.strip()
#     m = re.search(r"(19|20)\d{2}", ts)
#     if m:
#         return int(m.group(0))
#     try:
#         return datetime.fromisoformat(ts).year
#     except Exception:
#         return None


# def count_by_year(docs: Sequence[TimestampedDocument]) -> Dict[int, int]:
#     c: Dict[int, int] = defaultdict(int)
#     for d in docs:
#         y = parse_year(d.timestamp)
#         if y is not None:
#             c[y] += 1
#     return dict(c)


# def compute_cagr(yearly_count: Dict[int, int]) -> float:
#     if not yearly_count:
#         return 0.0
#     years = sorted(yearly_count)
#     if len(years) < 2:
#         return 0.0
#     start_year, end_year = years[0], years[-1]
#     periods = end_year - start_year
#     if periods <= 0:
#         return 0.0
#     start_val = max(1, yearly_count.get(start_year, 0))
#     end_val = max(0, yearly_count.get(end_year, 0))
#     if end_val <= 0:
#         return -1.0
#     return (end_val / start_val) ** (1 / periods) - 1


# def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
#     dot = sum(x * y for x, y in zip(a, b))
#     na = math.sqrt(sum(x * x for x in a)) or 1.0
#     nb = math.sqrt(sum(y * y for y in b)) or 1.0
#     return dot / (na * nb)


# def top_k_nearest_to_centroid(vectors: Sequence[Sequence[float]], top_k: int = 20) -> List[int]:
#     if not vectors:
#         return []
#     dim = len(vectors[0])
#     centroid = [0.0] * dim
#     for v in vectors:
#         for i, x in enumerate(v):
#             centroid[i] += x
#     n = len(vectors)
#     centroid = [x / n for x in centroid]

#     scored = [(i, cosine_similarity(v, centroid)) for i, v in enumerate(vectors)]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [i for i, _ in scored[:top_k]]



# __all__ = [
#     "TimestampedDocument",
#     "DomainScoredDocument",
#     "TrendMetrics",
#     "ExpertVote",
#     "ClusterInsight",
#     "FrontierIdentificationReport",
#     "DomainClassifier",
#     "TextEmbedder",
#     "Clusterer",
#     "ClusterNarrativeGenerator",
#     "ExpertJudge",
#     "CoreTextExtractor",
#     "SmallModelDomainClassifier",
#     "BGEM3Embedder",
#     "UmapHdbscanClusterer",
#     "ModelNarrativeGenerator",
#     "ModelExpertJudge",
#     "TrendAnalyzer",
#     "FrontierTechIdentificationAgent",
#     "IdentificationAgent",
#     "top_k_nearest_to_centroid",
#     "compute_cagr",
# ]