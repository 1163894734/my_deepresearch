# import math
# from typing import List, Dict
# from collections import defaultdict, Counter
# from smolagents import Tool
# import numpy as np
# import umap
# from hdbscan import HDBSCAN

# def cosine_similarity(a: List[float], b: List[float]) -> float:
#     dot = sum(x * y for x, y in zip(a, b))
#     na = math.sqrt(sum(x * x for x in a)) or 1.0
#     nb = math.sqrt(sum(y * y for y in b)) or 1.0
#     return dot / (na * nb)

# def top_k_nearest_to_centroid(vectors: List[List[float]], top_k: int = 20) -> List[int]:
#     if not vectors: return []
#     dim = len(vectors[0])
#     centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
#     scored = [(i, cosine_similarity(v, centroid)) for i, v in enumerate(vectors)]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [i for i, _ in scored[:top_k]]

# class ClusterDocumentsTool(Tool):
#     name = "cluster_documents"
#     description = "第二步：对保留下来的文献进行 BGE-M3 向量化和 UMAP+HDBSCAN 聚类，提取每个技术簇的核心文献。"
    
#     inputs = {
#         "kept_docs": {"type": "array", "description": "经过过滤后保留的文献字典列表"}
#     }
#     output_type = "array"

#     def __init__(self, model=None, **kwargs):
#         super().__init__()
#         from sentence_transformers import SentenceTransformer
#         self.embedder = SentenceTransformer("BAAI/bge-m3")

#     def forward(self, kept_docs: List[Dict]) -> List[Dict]:
#         if not kept_docs: return []
#         print(f"\n📥 [Cluster Skill] 正在向量化 {len(kept_docs)} 篇文献...")
        
#         # 1. 向量化
#         texts = [doc.get("core_text", "") for doc in kept_docs]
#         vectors = self.embedder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
#         vectors = [list(map(float, row)) for row in vectors]

#         # 2. 聚类
#         print("📥 [Cluster Skill] 正在执行 UMAP 降维与 HDBSCAN 聚类...")
#         reducer = umap.UMAP(n_components=10, metric="cosine", random_state=42, n_jobs=1, min_dist=0.0)
#         reduced = reducer.fit_transform(np.asarray(vectors, dtype=float))
#         labels = HDBSCAN(min_cluster_size=3, metric="euclidean").fit_predict(reduced)
        
#         by_cluster = defaultdict(list)
#         for idx, lb in enumerate(labels):
#             by_cluster[int(lb)].append(idx)
            
#         print(f"📤 [Cluster Skill] 聚类分布: {dict(Counter(labels))}")

#         # 3. 构建结果簇
#         clusters = []
#         for cluster_id, idxs in sorted(by_cluster.items(), key=lambda kv: kv[0]):
#             if cluster_id < 0: continue # 过滤噪声
            
#             cluster_vectors = [vectors[i] for i in idxs]
#             cluster_docs = [kept_docs[i] for i in idxs]
            
#             # 提取代表性文献
#             top_local_idx = top_k_nearest_to_centroid(cluster_vectors, top_k=min(20, len(cluster_docs)))
#             representatives = [cluster_docs[i] for i in top_local_idx]
            
#             clusters.append({
#                 "cluster_id": cluster_id,
#                 "size": len(cluster_docs),
#                 "representative_doc_ids": [d.get("doc_id", "") for d in representatives],
#                 "representative_summaries": [d.get("abstract") or d.get("core_text", "")[:260] for d in representatives],
#                 "cluster_docs": cluster_docs # 保留原始数据供下一步分析趋势
#             })
            
#         print(f"📤 [Cluster Skill] 成功提取 {len(clusters)} 个有效技术簇。")
#         return clusters