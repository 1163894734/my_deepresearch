import json
import os
import re
import ssl
import time
import ast
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import feedparser
from duckduckgo_search import DDGS
from sklearn.cluster import DBSCAN
from smolagents import Tool
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from smolagents import Tool
import utils.common_utils as common_utils
from utils.common_utils import ModelProvider
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import umap
from sklearn.preprocessing import normalize
# 1. 动态获取当前脚本 (custom_tools.py) 的绝对路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 往上一级推算到 open_deep_research 根目录，然后拼上 models 文件夹
project_root = os.path.dirname(os.path.dirname(current_script_dir))
models_dir = os.path.join(project_root, "models")

# 3. 动态拼装模型的绝对物理路径
embedder_path = os.path.join(models_dir, "bge-m3")
reranker_path = os.path.join(models_dir, "bge-reranker-base")
# 4. 加载模型（如果本地路径不存在，可以留个优雅的报错提示）
if not os.path.exists(embedder_path) or not os.path.exists(reranker_path):
    raise FileNotFoundError(f"❌ 找不到本地模型！请确保把模型下载并解压到了: {models_dir}")

EMBEDDER = SentenceTransformer(embedder_path, device='cpu') 
RERANKER = CrossEncoder(reranker_path, device='cpu')
# ==================== 聚类工具 (独立解耦) ====================
class TechnologyClustererTool(Tool):
    """
    自给自足的技术聚类工具。
    利用全局 BGE 模型进行语义向量化，结合 UMAP 降维与 HDBSCAN 密度聚类，识别出紧密关联的技术簇。
    """
    
    name = "technology_clusterer"
    description = "对结构化的学术/科技数据（字典列表）进行高维语义向量化、降维和密度聚类，发现潜在的前沿技术簇。算法会自动过滤掉孤立的噪声数据。"
    
    inputs = {
        "records": {
            "type": "array",
            "description": "需要聚类的原始数据列表，例如 [{'标题': '...', '创新点': '...'}, ...]"
        },
        "text_fields": {
            "type": "array",
            "description": "指定提取哪些字段拼接成文本用于计算语义向量。例如 ['标题', '关键词', '所属技术领域']。如果不传，自动提取关键字段。",
            "nullable": True
        }
    }
    
    output_type = "any" # 返回 List[List[Dict]] (聚类后的分组)

    def __init__(self):
        super().__init__()

    def forward(self, records: list, text_fields: Optional[list] = None) -> list:
        if not records or not isinstance(records, list):
            print("⚠️ 传入的聚类数据为空或格式不正确。")
            return []

        # 1. 动态提取特征文本
        texts_to_embed = []
        for rec in records:
            text_parts = []
            
            # 如果指定了提取字段
            if text_fields and isinstance(text_fields, list):
                for field in text_fields:
                    val = rec.get(field)
                    if val and isinstance(val, str):
                        text_parts.append(val)
            else:
                # 默认提取高浓度标签字段，避免长文本导致特征均值化
                default_fields = [
                    "标题", "关键词", "所属技术领域"
                ]
                for field in default_fields:
                    val = rec.get(field)
                    if val and isinstance(val, str):
                        text_parts.append(val)
                        
            # 兜底拼接
            if not text_parts:
                for val in rec.values():
                    if isinstance(val, str) and len(val.strip()) > 0:
                        text_parts.append(val)

            texts_to_embed.append(" ".join(text_parts))

        if not any(texts_to_embed):
            print("⚠️ 未提取到有效的文本特征，无法进行语义向量化。")
            return []

        print(f"🧠 正在调用 BGE 模型进行语义向量化 (共 {len(records)} 条数据)...")
        
        try:
            # 2. 计算语义向量并进行归一化
            vectors = EMBEDDER.encode(texts_to_embed, normalize_embeddings=True)
            
            # 3. UMAP 降维 (关键步骤：高维空间下密度聚类容易失效，降到 15-50 维最佳)
            print("🧠 正在使用 UMAP 进行向量降维优化...")
            reducer = umap.UMAP(
                n_neighbors=30,      # 关注局部结构（设小一点能分出更细的簇，建议 3-10）
                n_components=15,    # 降维后的维度
                metric='cosine',    # 文本特征使用余弦距离效果更好
                random_state=42     # 固定随机种子保证结果可复现
            )
            reduced_vectors = reducer.fit_transform(vectors)

            # reduced_vectors = vectors #测试一下去掉umap
            
            # 4. HDBSCAN 密度聚类
            print("🧠 正在使用 HDBSCAN 进行细粒度密度聚类...")
            # min_cluster_size 控制一个技术簇最少需要几篇论文
            # 动态调整：如果数据量很少，则降低门槛
            min_cluster_size = 10 if len(records) > 20 else 5
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size, 
                min_samples=2,                 # 控制对噪声的容忍度，越小越容易把边缘点拉入簇中
                metric='euclidean',            # 降维后使用欧氏距离
                cluster_selection_epsilon=0.1  # 允许合并极近的微小簇的距离阈值
            )
            labels = clusterer.fit_predict(reduced_vectors)

            # 5. 结果分组过滤
            clusters_dict = defaultdict(list)
            noise_count = 0
            for idx, label in enumerate(labels):
                if label >= 0:  # HDBSCAN 中 label 为 -1 表示孤立的噪声点，坚决抛弃
                    clusters_dict[label].append(records[idx])
                else:
                    noise_count += 1

            if not clusters_dict:
                print("⚠️ 聚类算法未发现明显密度簇，所有文献的话题过于发散，均被判定为孤立点。")
                return []

            print(f"✅ 聚类完成！共发现 {len(clusters_dict)} 个紧密的技术簇，果断剔除了 {noise_count} 条孤立数据的干扰。")
            
            # 6. 调用大模型为每一个聚类簇提取技术名称
            formatted_clusters = []
            print("🧠 正在调用大模型为聚类簇智能命名...")
            
            # 获取大模型实例
            model = ModelProvider.get_model()
            
            for cluster_id, papers in clusters_dict.items():
                # 将该簇内的论文标题和核心方法拼接，提供给大模型参考
                cluster_text_parts = []
                for p in papers:
                    title = p.get("标题", p.get("title", ""))
                    method = p.get("核心方法/方案总结", "")
                    if title:
                        cluster_text_parts.append(f"标题: {title}")
                    if method:
                        cluster_text_parts.append(f"方法: {method}")
                        
                # 截取适量文本防止 Token 超限
                cluster_context = "\n".join(cluster_text_parts)[:2000]
                
                system_prompt = "你是一个科技情报专家。请根据提供的文献标题和方法，总结出一个精准、专业的核心技术名称（只需输出纯文本名称，控制在15个字以内，绝对不要使用加粗、不要使用冒号，不要任何解释性文字）。"
                
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"文献内容如下：\n{cluster_context}"}
                    ]
                    response = model(messages)
                    cluster_name = response.content if hasattr(response, 'content') else str(response)
                    
                    # 清洗大模型输出，去除可能带有的多余符号
                    cluster_name = cluster_name.strip('`"\'。.,\n ')
                    
                    # 兜底机制 确保名称不为空
                    if not cluster_name:
                        cluster_name = f"技术簇_{cluster_id + 1}"
                        
                except Exception as e:
                    print(f"⚠️ 簇 {cluster_id + 1} 命名失败: {str(e)}")
                    cluster_name = f"技术簇_{cluster_id + 1}"

                formatted_clusters.append({
                    "cluster_label": cluster_name,
                    "papers": papers
                })
                
            print("✅ 智能命名完成！")
            return formatted_clusters
            
        except Exception as e:
            print(f"❌ 聚类过程发生底层错误: {str(e)}")
            return []

class AcademicSearchTool(Tool):
    """
    高精度、支持原生布尔逻辑的双引擎学术文献搜索工具 (支持 OpenAlex 和 arXiv)
    """
    name = "academic_search"
    description = "学术文献搜索工具。支持 'openalex' 和 'arxiv' 数据源。完美支持 AND, OR, NOT 及括号、双引号等布尔逻辑组合。"
    
    inputs = {
        "source": {
            "type": "string", 
            "description": "数据源，必须是 'openalex' 或 'arxiv'。"
        },
        "query": {
            "type": "string", 
            "description": "检索词。允许使用复杂的布尔逻辑（AND/OR/()）。工具会自动剥离如 TS= 的前缀标签。"
        },
        "per_page": {
            "type": "integer", 
            "description": "返回的文献数量，默认 10。", 
            "nullable": True,
            "default": 10
        },
    }
    output_type = "any"

    def __init__(self):
        super().__init__()
        self.last_arxiv_request_time = 0 

    def forward(self, source: str, query: str, per_page: int = 10) -> Dict[str, Any]:
        source = str(source).strip().lower()
        
        # ==========================================
        # 1. 柔性查询词清洗 (保留布尔语义)
        # ==========================================
        # 仅去除如 TS=, TI=, AB= 等特定数据库的字段标签，保留后面的内容
        clean_query = re.sub(r'\b[A-Za-z]{2,}\s*=\s*', '', query)
        # 整理多余的空格
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()

        if not clean_query:
            return {"error": "清洗后的检索词为空，请检查输入。"}

        print(f"🔍 触发检索 | 引擎: {source.upper()} | 原始词: [{query}]\n   ↳ 实际发送: [{clean_query}]")

        # ==========================================
        # 2. 路由到对应的搜索引擎
        # ==========================================
        if source == "openalex":
            return self._fetch_openalex(clean_query, per_page)
        elif source == "arxiv":
            return self._fetch_arxiv(clean_query, per_page)
        else:
            return {"error": f"不支持的数据源: '{source}'。请使用 'openalex' 或 'arxiv'。"}

    def _fetch_openalex(self, query: str, max_results: int) -> Dict[str, Any]:
        url = "https://api.openalex.org/works"
        # 声明邮箱进入 polite pool (礼貌池)，降低被限流的概率
        headers = {"User-Agent": "mailto:open_deep_research@example.com"}
        
        # 只取需要的字段，降低解析压力和网络开销
        select_fields = "id,doi,title,publication_year,authorships,abstract_inverted_index,cited_by_count"
        
        params = {
            "search": query, # OpenAlex 天然支持带括号和 AND/OR 的原生布尔表达式
            "per_page": min(max_results, 100),
            "select": select_fields,
            "sort": "relevance_score:desc"
        }

        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                
                if resp.status_code == 429:
                    print(f"⚠️ OpenAlex 限流，等待 {2 ** attempt} 秒后重试...")
                    time.sleep(2 ** attempt)
                    continue
                    
                resp.raise_for_status()
                data = resp.json()
                
                standardized_results = []
                for work in data.get("results", []):
                    # 还原倒排索引摘要
                    abstract_text = ""
                    inv_index = work.get("abstract_inverted_index")
                    if inv_index and isinstance(inv_index, dict):
                        word_positions = []
                        for word, positions in inv_index.items():
                            for pos in positions:
                                word_positions.append((pos, word))
                        word_positions.sort(key=lambda x: x[0])
                        abstract_text = " ".join([w for _, w in word_positions])

                    authors = []
                    for auth in work.get("authorships", []):
                        name = auth.get("author", {}).get("display_name")
                        if name:
                            authors.append(name)

                    standardized_results.append({
                        "id": work.get("id", ""),
                        "doi": work.get("doi", ""),
                        "title": work.get("title", "No Title"),
                        "year": str(work.get("publication_year", "")),
                        "authors": authors,
                        "abstract": abstract_text,
                        "citations": work.get("cited_by_count", 0),
                        "source": "OpenAlex"
                    })
                    
                return {
                    "meta": {"source": "openalex", "query": query, "count": len(standardized_results)},
                    "results": standardized_results
                }
                
            except Exception as e:
                if attempt == 2:
                    return {"error": f"OpenAlex API 请求失败: {str(e)}"}
                time.sleep(2)

    def _fetch_arxiv(self, query: str, max_results: int) -> Dict[str, Any]:
        # 强制冷却，满足 arXiv 的并发要求
        elapsed = time.time() - self.last_arxiv_request_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
            
        url = "http://export.arxiv.org/api/query"
        # 对于复杂的括号组合，arXiv API 有时解析会略显脆弱，但用 urlencode 包裹后大部分情况下兼容
        encoded_query = urllib.parse.quote(f"all:{query}")
        full_url = f"{url}?search_query={encoded_query}&start=0&max_results={min(max_results, 100)}&sortBy=relevance"

        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        for attempt in range(3):
            try:
                self.last_arxiv_request_time = time.time()
                resp = requests.get(full_url, timeout=15)
                
                if resp.status_code == 503:
                    print(f"⚠️ arXiv 防火墙拦截，等待 {5 * (attempt+1)} 秒后重试...")
                    time.sleep(5 * (attempt + 1))
                    continue
                    
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                
                standardized_results = []
                for entry in root.findall('atom:entry', namespaces):
                    title_elem = entry.find('atom:title', namespaces)
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else "No Title"
                    
                    summary_elem = entry.find('atom:summary', namespaces)
                    abstract = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else ""
                    
                    pub_elem = entry.find('atom:published', namespaces)
                    year = pub_elem.text[:4] if pub_elem is not None and pub_elem.text else ""
                    
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name_elem = author.find('atom:name', namespaces)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())

                    doi = ""
                    doi_elem = entry.find('arxiv:doi', namespaces)
                    if doi_elem is not None:
                        doi = f"https://doi.org/{doi_elem.text.strip()}"
                        
                    standardized_results.append({
                        "id": entry.find('atom:id', namespaces).text if entry.find('atom:id', namespaces) is not None else "",
                        "doi": doi,
                        "title": title,
                        "year": year,
                        "authors": authors,
                        "abstract": abstract,
                        "citations": 0,
                        "source": "arXiv"
                    })

                # return {
                #     "meta": {"source": "arxiv", "query": query, "count": len(standardized_results)},
                #     "results": standardized_results
                # }
                return standardized_results

            except ET.ParseError:
                return {"error": "arXiv 返回了无法解析的 XML 数据。"}
            except Exception as e:
                if attempt == 2:
                    return {"error": f"arXiv API 请求失败: {str(e)}"}
                time.sleep(3)
    


class LLMDataMappingTool(Tool):
    """
    异构数据智能映射与提取工具（支持并行批处理与指定数据类型）。
    输入一个原生 List 和指定的数据类型，利用大模型将其映射到规范的字段结构中，
    返回结构化的字典列表 (List[Dict])。
    """
    
    name = "heterogeneous_data_mapping"
    description = "将无结构/半结构的数据列表，通过大模型提取并映射为包含指定标准字段的结构化字典列表。必须明确指定目标数据类型（如'论文'、'专利'等）。支持并行批处理以提升效率。"
    
    inputs = {
        "raw_data_list": {
            "type": "array", 
            "description": "包含原始数据记录的列表，例如 [{'title': '...', 'abstract': '...'}]"
        },
        "target_type": {
            "type": "string",
            "description": "强制指定当前批次数据的目标类型。必须是以下之一：'论文', '专利', '项目', '报告', '网页'。"
        },
        "batch_size": {
            "type": "integer",
            "description": "每批并行处理的数据条数，默认为5",
            "nullable": True,
            "default": 5
        },
        "max_workers": {
            "type": "integer",
            "description": "最大并行线程数，默认为3",
            "nullable": True,
            "default": 3
        }
    }
    output_type = "any"

    # 定义各类型的必选字段结构
    TYPE_SCHEMAS = {
        "论文": "序号, 标题, 文献源名称, 作者及机构, 作者简介, 发表时间, 摘要, 关键词, 分类号, DOI, 所属技术领域, 引言, 核心方法/方案总结, 创新点, 结论, 资助信息, 引用量",
        "专利": "序号, 标题, 摘要, 申请人及单位, 公开号, 公开日, 申请号, 申请日, 专利类型, 公开国别, 权利要求核心点, 所属技术领域, 权利要求数量, 专利分类号",
        "项目": "序号, 项目名称, 项目类别, 研究目标, 主要研究内容, 主要研究方法, 所属技术领域, 项目经费, 研发周期, 项目成果, 项目编号, 资助机构, 承担人及机构, 项目阶段状态",
        "报告": "序号, 报告名称, 报告来源, 主要内容介绍, 主要观点梳理, 报告时间, 报告机构, 所属技术领域",
        "网页": "序号, 题目, 网页来源, 主要内容, 主要观点梳理, 发布时间, 发布人及机构, 所属技术领域"
    }

    def __init__(self):
        super().__init__()
        self.model = None
        self._lock = threading.Lock()

    def _get_model(self):
        """线程安全地获取模型实例"""
        if not self.model:
            with self._lock:
                if not self.model:
                    self.model = ModelProvider.get_model()
        return self.model

    def _process_batch(self, batch: List[Any], batch_idx: int, target_type: str, schema_str: str) -> List[Dict[str, Any]]:
        """处理单个批次的数据（在独立线程中执行）"""
        model = self._get_model()
        system_prompt = self._build_system_prompt(target_type, schema_str)
        
        user_prompt = f"请处理以下 {len(batch)} 条原始数据记录（批次 {batch_idx}）：\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
        
        print(f"🔄 [批次 {batch_idx}] 正在调用大模型处理 {len(batch)} 条数据，目标类型：{target_type}...")
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response_msg = model(messages)
            llm_output = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)
            
            structured_data = self._parse_json_like(llm_output)
            
            if structured_data and isinstance(structured_data, dict) and "results" in structured_data:
                results = structured_data["results"]
                if isinstance(results, list):
                    print(f"✅ [批次 {batch_idx}] 成功处理，返回 {len(results)} 条结果")
                    return results
            
            if structured_data and isinstance(structured_data, list):
                print(f"✅ [批次 {batch_idx}] 成功处理，返回 {len(structured_data)} 条结果")
                return structured_data
                
            print(f"⚠️ [批次 {batch_idx}] 返回格式异常，回退为错误记录")
            return [{"error": "大模型未返回有效的预期结构", "raw_output": llm_output[:500], "batch": batch_idx}]
            
        except Exception as e:
            print(f"❌ [批次 {batch_idx}] 处理失败: {str(e)}")
            return [{
                "error": f"批次 {batch_idx} 处理失败: {str(e)}",
                "original_data": item if isinstance(item, dict) else {"raw": item}
            } for item in batch]

    def _split_into_batches(self, data_list: List[Any], batch_size: int) -> List[List[Any]]:
        """将数据列表拆分成多个批次"""
        batches = []
        for i in range(0, len(data_list), batch_size):
            batches.append(data_list[i:i + batch_size])
        return batches

    def _merge_results(self, all_batch_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """合并所有批次的结果"""
        merged = []
        for batch_result in all_batch_results:
            if isinstance(batch_result, list):
                merged.extend(batch_result)
            else:
                merged.append({"error": "批次返回非列表结果", "data": batch_result})
        return merged

    def forward(self, raw_data_list: list, target_type: str, batch_size: int = 5, max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        并行处理异构数据映射
        """
        if not isinstance(raw_data_list, list) or not raw_data_list:
            return [{"error": "输入必须是一个非空的列表 (List)。"}]
            
        target_type = str(target_type).strip()
        if target_type not in self.TYPE_SCHEMAS:
            return [{"error": f"无效的 target_type: '{target_type}'。支持的类型为: {', '.join(self.TYPE_SCHEMAS.keys())}"}]
            
        schema_str = self.TYPE_SCHEMAS[target_type]
        
        print(f"🚀 开始并行处理，目标类型: [{target_type}]，总数据量: {len(raw_data_list)}，批次大小: {batch_size}")
        
        batches = self._split_into_batches(raw_data_list, batch_size)
        print(f"📦 共拆分为 {len(batches)} 个批次")
        
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_batch, batch, idx, target_type, schema_str): idx 
                for idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_idx):
                batch_idx = future_to_idx[future]
                try:
                    batch_results = future.result()
                    all_results.append(batch_results)
                except Exception as e:
                    print(f"💥 [批次 {batch_idx}] 未知异常: {str(e)}")
                    original_batch = batches[batch_idx]
                    all_results.append([{
                        "error": f"批次 {batch_idx} 执行异常: {str(e)}",
                        "original_data": item if isinstance(item, dict) else {"raw": item}
                    } for item in original_batch])
        
        merged_results = self._merge_results(all_results)
        print(f"🎉 处理完成！成功返回 {len(merged_results)} 条 [{target_type}] 结构化记录")
        
        return merged_results

    def _build_system_prompt(self, target_type: str, schema_str: str) -> str:
        """动态构建系统提示词，强制约束目标类型和对应的字段"""
        return f"""
你是一个顶级的科技情报数据分析专家。你的任务是将用户提供的非结构化或半结构化数据列表，逐条提取、总结，并严格按照 JSON 格式输出。

【重要前置条件】
用户已明确本次输入的数据类型统一为：**{target_type}**。
请你仅按照 {target_type} 的数据规范进行提取，不要擅自更改数据类型。

【工作流】
对于输入的每一条数据，你必须执行以下步骤：
1. 信息提取：提取原文中显式存在的对应字段信息。
2. 智能推断与总结：对于原文未直接给出但可以推断的字段（如"创新点"、"核心方法/方案总结"），请运用你的专业知识基于原文内容生成一两句话的高质量总结。
3. 缺失处理：如果某个字段完全无法提取且无法推断，请填入 null。

【数据结构规范】
你输出的必须是一个 JSON 对象 (Object)，该对象包含一个名为 "results" 的键，其值是一个列表 (List)。
列表中的每个对象代表一条处理后的数据，且须包含以下所有指定字段（不要遗漏字段，顺序无关）：
{schema_str}

【输出格式示例】
{{
  "results": [
    {{
        "字段1": "...",
        "字段2": "...",
        ...
    }}
  ]
}}

【输出要求】
只输出合法的 JSON，不要有任何 Markdown 标记（如 ```json），不要有任何解释性文字。
"""

    def _parse_json_like(self, text: str) -> Any:
        if not isinstance(text, str):
            return text
            
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except Exception:
            pass

        match = re.search(r"(\[.*\]|\{.*\})", clean, flags=re.DOTALL)
        if not match:
            return None
        try:
            return ast.literal_eval(match.group(0))
        except Exception:
            return None
# class ClusterSummarizerTool(Tool):
#     """
#     专门用于对技术簇内的多篇论文进行深度大模型总结，并强制输出结构化 JSON 的工具。
#     """
#     name = "cluster_summarizer"
#     description = "输入多篇论文的拼接文本，利用大模型提取并总结出该技术簇的名称、定义、痛点、原理、指标和价值，返回标准的字典格式。"
    
#     inputs = {
#         "text_content": {
#             "type": "string",
#             "description": "该聚类簇内所有论文的标题、摘要等拼接后的长文本纯文本内容"
#         }
#     }
#     output_type = "any"

#     def __init__(self):
#         super().__init__()

#     def forward(self, text_content: str) -> dict:
#         model = ModelProvider.get_model()
        
#         system_prompt = """
#         你是一个顶级的科技情报分析专家。请基于输入的文献摘要文本，进行深度总结。
        
#         【⚠️ 提取与生成规范】（最高优先级）
#         1. 强制中文：必须100%转化为专业、流畅的中文。禁止复制英文原文片段。
#         2. 凝练度要求：高度提炼，拒绝啰嗦背景。
#         3. 必须严格输出以下 JSON 格式（绝对不要输出 Markdown 标记，不要输出解释性文字）：
#         {
#             "技术名称": "精准代表核心技术（如：硅通孔(TSV)互连技术）",
#             "基本定义": "一句话概括核心技术或方向（50字以内）",
#             "拟解决痛点/问题": "直击缺陷或挑战（100字以内）",
#             "核心原理": "主要创新架构、机制或方法论（150字以内）",
#             "参数指标": "提取具体数值（如果没有则强制填'未提及具体量化指标'，严禁捏造）",
#             "作用价值": "核心贡献，精炼说明"
#         }
#         """
        
#         print("🧠 正在调用大模型对当前聚类簇进行深度阅读与总结...")
#         try:
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 # 截取前 6000 字符防止部分极端情况超长 token 爆掉
#                 {"role": "user", "content": f"请总结以下文献内容：\n{text_content[:6000]}"} 
#             ]
            
#             response = model(messages)
#             content = response.content if hasattr(response, 'content') else str(response)
            
#             # 简单清洗提取 JSON
#             clean = content.strip()
#             match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
#             if match:
#                 return json.loads(match.group(0))
#             else:
#                 return {"error": "JSON解析失败", "raw_output": content[:100]}
                
#         except Exception as e:
#             return {"技术名称": "总结生成失败", "error": str(e)}


import re

class TechSignalEvaluatorTool(Tool):
    """
    综合技术信号评估工具。
    同时识别前沿技术和弱信号，满足任一设定的阈值即可保留，并通过 signal_types 字段区分。
    为保护下游大模型，输出时每个簇最多保留最新的 20 篇文献。
    """
    
    name = "tech_signal_evaluator"
    description = "对聚类簇同时进行前沿技术和弱信号的判定。返回符合任一特征的聚类簇，内部包含 signal_types 字段标明其类型，以及具体满足的条件列表。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "包含多个聚类簇的列表，每个簇需包含 cluster_label 和 papers 字段。"
        },
        "frontier_threshold": {
            "type": "integer",
            "description": "判定为前沿技术所需满足的最少条件数，默认为 2。",
            "nullable": True,
            "default": 2
        },
        "weak_threshold": {
            "type": "integer",
            "description": "判定为弱信号所需满足的最少条件数，默认为 2。",
            "nullable": True,
            "default": 2
        }
    }
    
    output_type = "any"

    def __init__(self):
        super().__init__()
        self.current_year = 2026
        # 量化探测规则合并
        self.metric_pattern = re.compile(r'(\d+\.?\d*\s*(?:%|ms|s|m|w|dB|dBm|°c|fps|gb|tb|nm|μm|Ω|mol/l|ppm))', re.IGNORECASE)
        self.application_keywords = [
            "应用", "潜力", "前景", "价值", "落地", "场景", "商用", "产业化",
            "application", "potential", "value", "prospect", "commercial"
        ]
        self.exploration_keywords = [
            "概念", "初步验证", "理论探索", "实验室阶段", "初期", "探索性", "设想", "提议",
            "concept", "preliminary", "theoretical", "initial stage", "exploration", "proposal"
        ]
        self.novelty_keywords = [
            "新原理", "新交叉", "新方向", "新范式", "前所未有", "新兴", "潜在方向",
            "novel principle", "emerging", "new paradigm", "new direction", "unprecedented", "intersection"
        ]

    def _extract_institutions(self, authors_list):
        institutions = set()
        
        # 🛡️ 兼容防爆1：如果传进来的直接是字符串（比如 "张三, 李四"），先套成列表
        if isinstance(authors_list, str):
            authors_list = [authors_list]
        elif isinstance(authors_list, dict): # 🛡️ 兼容防爆2：万一是单个字典
            authors_list = [authors_list]
        elif not isinstance(authors_list, list):
            return institutions
            
        for author_info in authors_list:
            if not author_info:
                continue
                
            # 🌟 核心防爆修改：不管大模型给的是字符串还是字典，一律强转成字符串再正则！
            author_str = str(author_info)
            
            match = re.search(r'\((.*?)\)', author_str)
            if match:
                insts = match.group(1).split(',')
                for inst in insts:
                    institutions.add(inst.strip())
            else:
                parts = re.split(r'[,;]', author_str)
                if len(parts) > 1:
                    institutions.add(parts[-1].strip())
                else:
                    institutions.add(author_str.strip())
        return institutions

    def _extract_year(self, pub_time):
        match = re.search(r'\b(19\d{2}|20\d{2})\b', str(pub_time))
        if match:
            return int(match.group(1))
        return None

    def forward(self, clusters_data: list, frontier_threshold: int = 2, weak_threshold: int = 2) -> list:
        if not isinstance(clusters_data, list):
            return [{"error": "输入必须是聚类簇的列表"}]
        
        print(f"🔬 正在对 {len(clusters_data)} 个聚类簇进行综合信号评估 (前沿阈值 {frontier_threshold}，弱信号阈值 {weak_threshold} )...")
        
        evaluated_clusters = []
        
        for cluster in clusters_data:
            cluster_label = cluster.get("cluster_label", "Unknown")
            papers = cluster.get("papers", [])
            
            if not papers:
                continue

            # 特征统计变量初始化 (基于完整文献列表，保证判定准确)
            vol = len(papers)
            years = []
            has_metrics = False
            has_app_value = False
            is_exploration = False
            is_novelty = False
            all_institutions = set()
            valid_methods_count = 0
            
            for paper in papers:
                abstract = paper.get("摘要", "") or ""
                method = paper.get("核心方法/方案总结", "") or ""
                conclusion = paper.get("结论", "") or ""
                authors = paper.get("作者及机构", []) or []
                pub_time = paper.get("发表时间", "") or ""
                
                combined_text = (abstract + method + conclusion).lower()

                if self.metric_pattern.search(combined_text):
                    has_metrics = True
                if any(kw in combined_text for kw in self.application_keywords):
                    has_app_value = True
                if any(kw in combined_text for kw in self.exploration_keywords):
                    is_exploration = True
                if any(kw in combined_text for kw in self.novelty_keywords):
                    is_novelty = True

                all_institutions.update(self._extract_institutions(authors))

                year = self._extract_year(pub_time)
                if year:
                    years.append(year)
                    
                if method:
                    valid_methods_count += 1

            # 前沿技术条件判定
            frontier_conditions = []
            if vol >= 3:
                frontier_conditions.append(f"文献数量较多(共{vol}篇)，形成明显技术簇")
            if has_metrics and valid_methods_count > 0:
                frontier_conditions.append("技术具备清晰的定义、方法、实验及具体指标")
            if years:
                recent_count = sum(1 for y in years if self.current_year - y <= 2)
                recent_ratio = recent_count / len(years)
                if recent_ratio >= 0.5: 
                    frontier_conditions.append(f"近1-2年发文量增长(近两年占比{recent_ratio:.0%})，热度呈上升趋势")
            if has_app_value:
                frontier_conditions.append("技术具有潜在的落地场景与实际应用价值")
            inst_count = len(all_institutions)
            if inst_count >= 2:
                frontier_conditions.append(f"有多个({inst_count}个)研究机构跟进研究")

            # 弱信号条件判定
            weak_conditions = []
            if vol <= 2:
                weak_conditions.append(f"研究文献数量极少(共{vol}篇)，尚未形成技术簇")
            if is_exploration:
                weak_conditions.append("以概念探索、初步验证为主，处于理论探索或实验室初期，技术路线不明确")
            if inst_count <= 1:
                weak_conditions.append(f"暂无明显研究热度，未被主流关注(仅{inst_count}个机构参与)")
            if is_novelty:
                weak_conditions.append("属于新原理、新交叉、新方向，未来潜力待时间检验")

            # 综合归类
            is_frontier = len(frontier_conditions) >= frontier_threshold
            is_weak = len(weak_conditions) >= weak_threshold
            
            signal_types = []
            if is_frontier:
                signal_types.append("前沿技术")
            if is_weak:
                signal_types.append("弱信号")

            if signal_types:
                # 克隆原数据，防止污染
                cluster_copy = cluster.copy()
                
                # ==========================================
                # 🚀 核心防爆修改：截取最新的 20 篇文献传递给下游
                # ==========================================
                def get_paper_year(p):
                    y = self._extract_year(p.get("发表时间", ""))
                    return y if y else 0
                
                # 按时间降序排序
                sorted_papers = sorted(papers, key=get_paper_year, reverse=True)
                
                # 截取前20篇
                cluster_copy["papers"] = sorted_papers[:20]
                
                # 增加终端提示，让我们知道哪几个簇被截断了
                if len(papers) > 20:
                    print(f"   🛡️ 触发防爆机制：簇[{cluster_label}] 有 {len(papers)} 篇文献，已截断保留最新的 20 篇供下游分析。")
                # ==========================================

                cluster_copy["signal_types"] = signal_types
                
                # 保留详细判定理由
                if is_frontier:
                    cluster_copy["frontier_conditions"] = frontier_conditions
                if is_weak:
                    cluster_copy["weak_signal_conditions"] = weak_conditions
                    
                # 合并所有命中的条件到一个兼容字段中
                cluster_copy["met_conditions"] = frontier_conditions + weak_conditions
                
                evaluated_clusters.append(cluster_copy)
                
                print(f"🌟 识别成功 [{cluster_label}] 属于 {' + '.join(signal_types)}")
                if is_frontier:
                    print(f"   前沿条件满足 {len(frontier_conditions)} 条")
                if is_weak:
                    print(f"   弱信号条件满足 {len(weak_conditions)} 条")
            else:
                print(f"❌ 淘汰普通簇 [{cluster_label}]")

        print(f"✅ 综合评估完成，共筛选出 {len(evaluated_clusters)} 个价值目标。")
        return evaluated_clusters

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from smolagents import Tool
from utils.common_utils import ModelProvider

class TechConnotationExtractorTool(Tool):
    name = "tech_connotation_extractor"
    description = "对过滤后的技术簇进行内涵提取 利用大模型阅读簇内论文 提取基本定义 痛点 原理 体系 指标和价值6个维度。采用Map-Reduce架构解决长文本截断引发的幻觉。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "经过评估工具输出的包含 cluster_label 和 papers 等字段的聚类簇列表"
        }
    }
    
    output_type = "any"

    def __init__(self):
        super().__init__()

    def _map_single_paper(self, paper: dict, model) -> str:
        """【Map阶段】: 独立提取单篇文献的核心要素，防止后期信息过载"""
        title = paper.get("标题", "")
        abstract = paper.get("摘要", "")
        method = paper.get("核心方法/方案总结", "")
        
        prompt = f"""
        请严格基于以下文献，提取核心技术要素，总字数控制在150字以内，必须是纯文本：
        标题：{title}
        摘要：{abstract}
        方法：{method}
        
        请提取：1. 主要痛点/面临问题 2. 核心原理/提出方案 3. 提及的具体量化指标(若原文未提及数值，请务必写"无")
        """
        
        try:
            response = model([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            return f"《{title}》提炼特征：\n{content}"
        except Exception as e:
            return f"《{title}》提炼失败：{str(e)}"

    def forward(self, clusters_data: list) -> list:
        if not isinstance(clusters_data, list):
            return [{"error": "输入必须是列表格式"}]
            
        if not clusters_data:
            print("⚠️ 输入的簇列表为空，无需提取")
            return []

        model = ModelProvider.get_model()
        updated_clusters = []
        
        print(f"🚀 正在调用大模型(Map-Reduce模式)对 {len(clusters_data)} 个技术簇进行内涵提取...")

        system_prompt = """
        你是一个顶级的科技情报分析专家。请基于以下汇总的【多篇同簇文献核心要素提炼】，进行深度的情报提取。
        
        【⚠️ 提取与生成规范】（最高优先级）
        1. 强制中文：必须100%转化为专业流畅的学术中文。
        2. 纯文本无标记：生成的内容必须是纯文本段落。绝对禁止在文本内部使用加粗和冒号等标记。
        3. 事实对齐：绝对禁止编造未在输入提炼中出现的指标或数值。若无具体指标，强制填"未提及具体量化指标"。
        4. 必须严格输出以下 JSON 格式 绝对不要输出 Markdown 标记 不要输出任何解释性文字：
        {
            "基本定义": "一句话精准概括该技术是什么 50字以内",
            "拟解决科学问题/痛点": "明确指出传统方法存在的缺陷或该技术致力于攻克的难题",
            "核心原理": "提炼该技术的主要创新架构机制或方法论核心",
            "技术体系": "该技术涉及的关键组件技术流派或上下游环节",
            "技术指标": "提取具体的数值或量化指标参数 如果没有明确数字强制填 未提及具体量化指标",
            "作用价值": "总结该技术对实际应用或未来发展的核心贡献"
        }
        """

        for idx, cluster in enumerate(clusters_data):
            cluster_label = cluster.get("cluster_label", f"未知簇_{idx}")
            papers = cluster.get("papers", [])
            
            print(f"  > 正在抽取 [{cluster_label}] 的技术内涵 (共 {len(papers)} 篇文献)...")
            
            # ==========================================
            # 1. Map 阶段：并发处理该簇内的所有文献
            # ==========================================
            mapped_results = []
            print(f"    - 启动 Map 阶段：并发压缩 {len(papers)} 篇文献...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_paper = {executor.submit(self._map_single_paper, p, model): p for p in papers}
                for future in as_completed(future_to_paper):
                    try:
                        res = future.result()
                        mapped_results.append(res)
                    except Exception as e:
                        print(f"    - 单篇文献 Map 提取失败: {e}")
            
            # ==========================================
            # 2. Reduce 阶段：将所有浓缩后的特征拼接归纳
            # ==========================================
            reduced_context = "\n\n".join(mapped_results)
            # 增加一个宽松的兜底防爆保护，通常经过 Map 后 50 篇文献也不会超过 12000 字符
            reduced_context = reduced_context[:16000] 
            
            updated_cluster = cluster.copy()
            print(f"    - 启动 Reduce 阶段：全局内涵聚合...")

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"以下是该技术簇内文献的特征提炼汇总：\n{reduced_context}"} 
                ]
                
                response = model(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # --- 🌟 保留你强大的 JSON 清洗逻辑 ---
                clean = content.strip()
                if "```json" in clean:
                    clean = clean.split("```json")[1].split("```")[0].strip()
                elif "```" in clean:
                    parts = clean.split("```")
                    if len(parts) >= 3:
                        clean = parts[1].strip()
                
                try:
                    start_idx = clean.find('{')
                    if start_idx != -1:
                        clean = clean[start_idx:]
                        decoder = json.JSONDecoder()
                        result_dict, _ = decoder.raw_decode(clean)
                    else:
                        result_dict = {}
                except Exception as e:
                    print(f"    - JSON 解析底层失败: {e}")
                    result_dict = {}
                # ------------------------------------
                
                expected_keys = ["基本定义", "拟解决科学问题/痛点", "核心原理", "技术体系", "技术指标", "作用价值"]
                for key in expected_keys:
                    updated_cluster[key] = result_dict.get(key, "模型未能按要求生成此字段，原因为输入特征不明显或解析异常")

            except Exception as e:
                print(f"❌ 抽取失败 {str(e)}")
                for key in ["基本定义", "拟解决科学问题/痛点", "核心原理", "技术体系", "技术指标", "作用价值"]:
                    updated_cluster[key] = f"抽取异常 {str(e)}"

            updated_clusters.append(updated_cluster)

        print("✅ 内涵提取完成！已将核心字段成功注入原数据结构 (Map-Reduce 正常)。")
        return updated_clusters
    
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from smolagents import Tool
from utils.common_utils import ModelProvider

class TechEvolutionAnalyzerTool(Tool):
    name = "tech_evolution_analyzer"
    description = "对技术簇进行全面的演进态势分析。利用大模型提取主流路线、难题、方案、动向、趋势及演化图。采用 Map-Reduce 架构彻底消除长文本截断。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "包含 papers 字段的聚类簇列表数据。"
        }
    }
    
    output_type = "any"

    def __init__(self):
        super().__init__()

    def _map_single_paper(self, paper: dict, model) -> str:
        """【Map阶段】: 专门针对技术演进，提取时间特征和突破口"""
        title = paper.get("标题", "")
        method = paper.get("核心方法/方案总结", "")
        # 加入年份特征，方便后期 Reduce 阶段梳理演化图
        year = str(paper.get("发表时间", "未知年份"))
        
        prompt = f"""
        请从以下文献中提取与【技术演进、路线突破、未来动向】相关的信息，忽略背景废话，总字数控制在100字以内：
        标题：{title} (发表时间: {year})
        方案/方法：{method}
        """
        try:
            response = model([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            return f"【{year}】《{title}》提取：\n{content}"
        except Exception as e:
            return f"《{title}》提炼失败：{str(e)}"

    def forward(self, clusters_data: list) -> list:
        if not isinstance(clusters_data, list):
            return [{"error": "输入必须是列表格式"}]
            
        if not clusters_data:
            print("⚠️ 输入的簇列表为空，无需分析。")
            return []

        model = ModelProvider.get_model()
        updated_clusters = []
        
        print(f"🧠 正在调用大模型(Map-Reduce模式)对 {len(clusters_data)} 个技术簇进行【技术演进路径分析】...")

        system_prompt = """
        你是一个顶级的科技情报与技术战略分析专家。请基于输入的【按时间维度的文献提炼汇总】，进行技术演进态势的深度结构化提取。
        
        【⚠️ 提取与生成规范】（最高优先级）
        1. 强制中文：必须100%转化为专业流畅的学术与战略情报中文。
        2. 纯文本无标记：生成的内容必须是纯文本。绝对禁止在文本内部使用加粗（**）和冒号（：）等Markdown标记。
        3. 事实对齐：严格基于提供的时间线和突破进行总结，不可凭空捏造未提及的趋势。
        4. 必须严格输出以下 JSON 格式（绝对不要输出任何解释性文字）：
        {
            "主流技术路线": "概括当前该领域最主要的技术发展方向或演进脉络",
            "难题瓶颈": "指出该技术路线在进一步发展中遭遇的物理、工程、材料或成本层面的核心阻碍",
            "实现方案": "总结文献中提出的用以突破上述瓶颈的具体实施路径或解决方案",
            "新进发展动向": "提取文献中反映出的最新突破、新方法的引入或近期研究的焦点转移",
            "未来发展趋势": "基于当前研究，预测该技术在未来3至5年的演进方向与潜在应用扩展",
            "技术演化图": "用纯文本关系链的形式描绘技术的迭代过程（例如 早期技术A -> 过渡技术B -> 当前技术C -> 探索性分支D），不要使用特殊绘图字符"
        }
        """

        for idx, cluster in enumerate(clusters_data):
            cluster_label = cluster.get("cluster_label", f"未知簇_{idx}")
            papers = cluster.get("papers", [])
            
            print(f"  > 正在分析 [{cluster_label}] 的演进态势 (共 {len(papers)} 篇文献)...")
            
            # ==========================================
            # 1. Map 阶段
            # ==========================================
            mapped_results = []
            print(f"    - 启动 Map 阶段：提炼 {len(papers)} 篇文献演进特征...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_paper = {executor.submit(self._map_single_paper, p, model): p for p in papers}
                for future in as_completed(future_to_paper):
                    try:
                        mapped_results.append(future.result())
                    except Exception as e:
                        print(f"    - 单篇提取异常: {e}")
            
            # 按照年份前缀（如【2024】）自然排序，极大地帮助大模型在Reduce阶段梳理演化链条
            mapped_results.sort()
            reduced_context = "\n\n".join(mapped_results)
            reduced_context = reduced_context[:16000]

            updated_cluster = cluster.copy()
            print(f"    - 启动 Reduce 阶段：全局态势聚合...")

            try:
                # ==========================================
                # 2. Reduce 阶段
                # ==========================================
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请深度分析以下文献的演进时间线：\n{reduced_context}"} 
                ]
                
                response = model(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # --- 🌟 保留你强大的 JSON 清洗逻辑 ---
                clean = content.strip()
                if "```json" in clean:
                    clean = clean.split("```json")[1].split("```")[0].strip()
                elif "```" in clean:
                    parts = clean.split("```")
                    if len(parts) >= 3:
                        clean = parts[1].strip()
                
                try:
                    start_idx = clean.find('{')
                    if start_idx != -1:
                        clean = clean[start_idx:]
                        decoder = json.JSONDecoder()
                        result_dict, _ = decoder.raw_decode(clean)
                    else:
                        result_dict = {}
                except Exception as e:
                    print(f"    - JSON 解析底层失败: {e}")
                    result_dict = {}
                # ------------------------------------
                
                expected_keys = [
                    "主流技术路线", "难题瓶颈", "实现方案", 
                    "新进发展动向", "未来发展趋势", "技术演化图"
                ]
                for key in expected_keys:
                    updated_cluster[key] = result_dict.get(key, "模型未能按要求生成此字段")

            except Exception as e:
                print(f"❌ [{cluster_label}] 分析失败 {str(e)}")
                for key in ["主流技术路线", "难题瓶颈", "实现方案", "新进发展动向", "未来发展趋势", "技术演化图"]:
                    updated_cluster[key] = f"分析异常 {str(e)}"

            updated_clusters.append(updated_cluster)

        print("✅ 演进态势分析完成！已将所有扩展字段成功注入原数据结构 (Map-Reduce 正常)。")
        return updated_clusters
class FrontierReportGeneratorTool(Tool):
    """
    关键前沿技术识别与分析报告生成工具。
    利用前面步骤提取的精华字段和少量代表性文献，防止上下文爆炸。
    """
    
    name = "frontier_report_generator"
    description = "基于技术内涵与演进分析结果，提炼并生成《关键前沿技术识别与分析报告》(Markdown格式)。自动选取每个前沿簇的核心特征与代表性文献进行综合分析。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "经过评估和内涵/演进提取的聚类簇列表。"
        }
    }
    
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, clusters_data: list) -> str:
        if not isinstance(clusters_data, list):
            return "错误：输入必须是列表格式"
            
        # 1. 过滤出被判定为“前沿技术”的簇
        frontier_clusters = [c for c in clusters_data if "前沿技术" in c.get("signal_types", [])]
        
        if not frontier_clusters:
            print("⚠️ 未发现前沿技术簇，生成空报告。")
            return "# 关键前沿技术识别与分析报告\n\n本次深度分析未发现符合前沿技术阈值标准的集群。"

        print(f"📝 正在为 {len(frontier_clusters)} 个前沿技术簇撰写综合报告...")

        # 2. 组装极简的高质量上下文 (防爆核心逻辑)
        context_parts = []
        for idx, cluster in enumerate(frontier_clusters):
            label = cluster.get("cluster_label", f"前沿技术_{idx+1}")
            
            # 提取现成的高密度精华结论
            definition = cluster.get("基本定义", "暂无")
            pain_point = cluster.get("拟解决科学问题/痛点", "暂无")
            principle = cluster.get("核心原理", "暂无")
            tech_system = cluster.get("技术体系", "暂无")
            metrics = cluster.get("技术指标", "暂无")
            value = cluster.get("作用价值", "暂无")
            evolution = cluster.get("主流技术路线", "暂无")
            bottleneck = cluster.get("难题瓶颈", "暂无")
            solutions = cluster.get("实现方案", "暂无")
            trends = cluster.get("未来发展趋势", "暂无")
            evolution_map = cluster.get("技术演化图", "暂无")
            
            # 提取代表性文献 (按时间排序取前3篇)
            papers = cluster.get("papers", [])
            def get_year(p):
                y = p.get("发表时间", "")
                match = re.search(r'\b(20\d{2})\b', str(y))
                return int(match.group(1)) if match else 0
                
            sorted_papers = sorted(papers, key=get_year, reverse=True)
            top_papers = sorted_papers[:3] # 只取前3篇防爆
            paper_str = "\n".join([f"    - 《{p.get('标题', '未知标题')}》 ({p.get('发表时间', '未知年份')}): {p.get('核心方法/方案总结', '暂无方法信息')}" for p in top_papers])

            # 拼装给大模型的精简提示词片段
            cluster_context = f"### 技术方向 {idx+1}: {label}\n"
            cluster_context += f"- **基本定义**: {definition}\n"
            cluster_context += f"- **核心痛点与瓶颈**: {pain_point} | {bottleneck}\n"
            cluster_context += f"- **核心原理与体系**: {principle} | {tech_system}\n"
            cluster_context += f"- **解决与实现方案**: {solutions}\n"
            cluster_context += f"- **技术指标与价值**: {metrics} | {value}\n"
            cluster_context += f"- **演进路线与趋势**: {evolution} | {trends}\n"
            cluster_context += f"- **技术演化图谱**: {evolution_map}\n"
            cluster_context += f"- **代表性先锋文献(Top 3)**:\n{paper_str}\n"
            
            context_parts.append(cluster_context)

        final_context = "\n\n".join(context_parts)

        # 3. 调用大模型撰写最终报告
        model = ModelProvider.get_model()
        system_prompt = """你是一位供职于全球顶尖科技智库的首席微电子情报分析师。你需要根据传入的 JSON 格式技术簇数据，撰写一份高度严谨的《关键前沿技术识别与分析报告》。

        【🔴 绝对红线（违反将导致任务失败）】
        1. 严禁虚构：你引用的所有论文标题、作者、年份、期刊、数据参数必须 100% 来源于提供的 JSON 数据。绝对不可调用自身知识库编造文献！
        2. 严谨表述：禁用“彻底颠覆”、“革命性”、“完美”等夸大且绝对化的词汇。请使用“有望实现”、“潜在方向”、“显著提升”等客观学术用语。

        【📋 结构与内容强制要求】
        1. 宏观架构合并：传入的数据可能有二三十个细分簇。请不要流水账式地罗列！你必须将这些细分簇归纳为 3 到 5 个【宏观核心技术领域】（例如：先进材料与界面工程、单片三维集成架构、近传感与存算一体、前沿微纳封装与互连等），在每个大领域下再阐述细分技术。
        2. 强制学术引用：在文中提及任何技术突破时，必须以内文引用的方式标注来源，格式为：“......（作者名, 年份, 文献源名称）”。
        3. 强数据支撑：每个技术领域必须提取并列出具体的性能指标、工艺温度、良率或测试数据（如 <300°C、2.4 cm²/V·s 等），禁止纯定性描述。
        4. TRL 与风险评估：在每个宏观技术领域末尾，必须给出预估的【技术成熟度 TRL (1-9级)】，并客观指出当前的【核心工程挑战/量产瓶颈】。
        5. 数据可视化（Markdown图表）：
        - 必须在报告中生成至少 2 个 Markdown 表格（例如：不同技术路线的性能对比表、关键文献与核心指标汇总表）。
        - 尝试在总结部分，使用 Mermaid 语法 (` ```mermaid ... ``` `) 绘制一张宏观的【技术演进路线图 (Roadmap)】。

        请以极其专业的、适合国家级基金申报和正式学术研讨的严谨风格输出完整的 Markdown 报告。报告应包含：执行摘要、宏观技术图谱详解（含图表与TRL）、核心瓶颈与破局建议。
        """

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"以下是高度提炼的前沿技术核心数据，请据此撰写完整的深度分析报告：\n\n{final_context}"} 
            ]
            response = model(messages)
            report_content = response.content if hasattr(response, 'content') else str(response)
            print("✅ 《关键前沿技术识别与分析报告》撰写成功！")
            return report_content.strip()
        except Exception as e:
            print(f"❌ 前沿报告撰写失败: {str(e)}")
            return f"# 报告生成失败\n内部错误信息: {str(e)}"
        
class WeakSignalReportGeneratorTool(Tool):
    """
    弱信号技术识别与分析报告生成工具。
    专门针对具有“弱信号”特征的集群进行颠覆性潜力与萌芽状态分析。
    """
    
    name = "weak_signal_report_generator"
    description = "基于技术内涵与演进分析结果，提炼并生成《弱信号技术识别与分析报告》(Markdown格式)。重点挖掘和分析早期探索性技术的颠覆性潜力。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "经过评估和内涵/演进提取的聚类簇列表。"
        }
    }
    
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, clusters_data: list) -> str:
        if not isinstance(clusters_data, list):
            return "错误：输入必须是列表格式"
            
        # 1. 过滤出被判定为“弱信号”的簇
        weak_clusters = [c for c in clusters_data if "弱信号" in c.get("signal_types", [])]
        
        if not weak_clusters:
            print("⚠️ 未发现弱信号技术簇，生成空报告。")
            return "# 弱信号技术识别与分析报告\n\n本次深度分析未发现处于早期探索阶段的潜在颠覆性弱信号技术。"

        print(f"📝 正在为 {len(weak_clusters)} 个弱信号技术簇撰写综合报告...")

        # 2. 组装极简的高质量上下文
        context_parts = []
        for idx, cluster in enumerate(weak_clusters):
            label = cluster.get("cluster_label", f"弱信号技术_{idx+1}")
            
            definition = cluster.get("基本定义", "暂无")
            novelty = cluster.get("新进发展动向", "暂无")
            principle = cluster.get("核心原理", "暂无")
            value = cluster.get("作用价值", "暂无")
            bottleneck = cluster.get("难题瓶颈", "暂无")
            evolution_map = cluster.get("技术演化图", "暂无")
            
            papers = cluster.get("papers", [])
            def get_year(p):
                y = p.get("发表时间", "")
                match = re.search(r'\b(20\d{2})\b', str(y))
                return int(match.group(1)) if match else 0
                
            sorted_papers = sorted(papers, key=get_year, reverse=True)
            top_papers = sorted_papers[:3]
            paper_str = "\n".join([f"    - 《{p.get('标题', '未知标题')}》 ({p.get('发表时间', '未知年份')}): {p.get('核心方法/方案总结', '暂无方法信息')}" for p in top_papers])

            cluster_context = f"### 潜在颠覆性方向 {idx+1}: {label}\n"
            cluster_context += f"- **基本定义**: {definition}\n"
            cluster_context += f"- **萌芽新动向(新颖性)**: {novelty}\n"
            cluster_context += f"- **核心原理突破**: {principle}\n"
            cluster_context += f"- **潜在颠覆性价值**: {value}\n"
            cluster_context += f"- **当前面临阻碍**: {bottleneck}\n"
            cluster_context += f"- **演进路径预判**: {evolution_map}\n"
            cluster_context += f"- **早期先驱性文献(Top 3)**:\n{paper_str}\n"
            
            context_parts.append(cluster_context)

        final_context = "\n\n".join(context_parts)

        # 3. 调用大模型撰写报告
        model = ModelProvider.get_model()
        system_prompt = """你是一位供职于全球顶尖科技智库的首席前沿交叉科学分析师。你需要根据传入的 JSON 格式弱信号技术数据，撰写一份高度严谨的《弱信号技术识别与分析报告》。

        【🔴 绝对红线】
        1. 真实溯源：引用的论点、作者、期刊及实验数据必须严格取自提供的 JSON，严禁 AI 虚构或“脑补”文献。
        2. 客观克制：弱信号代表处于极早期（概念或实验室验证阶段）的技术。必须明确其高不确定性，禁用“即将爆发”、“即将取代”等绝对化断言。

        【📋 结构与内容强制要求】
        1. 主题凝练：不要罗列散碎的弱信号簇。请将它们聚合成 3-4 个【前瞻概念象限】（如：非传统计算原语、异构空间新材料、亚波长感知与干预等）。
        2. 学术规范锚定：必须使用标准的文献引用格式（如：“正如某某团队(年份, 期刊)在研究中指出的...”）。
        3. TRL 与转化路径：明确指出这些弱信号目前的 TRL（通常在 1-3 级），并分析其从“弱信号”向“前沿主流技术”演进所需的【关键触发条件】（如某项材料的突破、某类设备的问世）。
        4. 对比与演进表格：必须输出 1-2 个 Markdown 表格，对比“传统技术路线”与这些“弱信号颠覆性路线”在原理、理论极限上的核心差异。

        输出风格需兼具“极强的战略前瞻性”与“极严谨的学术实证感”，直接达到可作为国家级中长期科技规划草案的标准。
        """

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"以下是提取出的弱信号技术核心数据，请据此撰写深度报告：\n\n{final_context}"} 
            ]
            response = model(messages)
            report_content = response.content if hasattr(response, 'content') else str(response)
            print("✅ 《弱信号技术识别与分析报告》撰写成功！")
            return report_content.strip()
        except Exception as e:
            print(f"❌ 弱信号报告撰写失败: {str(e)}")
            return f"# 报告生成失败\n内部错误信息: {str(e)}"