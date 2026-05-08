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
EMBEDDER = SentenceTransformer('BAAI/bge-m3', device='mps')
RERANKER = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
# ==================== 聚类工具 (独立解耦) ====================
class TechnologyClustererTool(Tool):
    """
    自给自足的技术聚类工具。
    利用全局 BGE 模型进行语义向量化，结合 DBSCAN 密度聚类，识别出紧密关联的技术簇。
    """
    
    name = "technology_clusterer"
    description = "对结构化的学术/科技数据（字典列表）进行高维语义向量化和密度聚类，发现潜在的前沿技术簇。算法会自动过滤掉孤立的噪声数据。"
    
    inputs = {
        "records": {
            "type": "array",
            "description": "需要聚类的原始数据列表，例如 [{'标题': '...', '创新点': '...'}, ...]"
        },
        "text_fields": {
            "type": "array",
            "description": "指定提取哪些字段拼接成文本用于计算语义向量。例如 ['核心方法/方案总结', '创新点', '摘要']。如果不传，自动提取关键字段。",
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
                # 默认提取你刚才异构映射产生的高价值字段
                default_fields = [
                    "核心方法/方案总结", "创新点", "标题"
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
            # 2. 计算语义向量并进行 L2 归一化 (使用你全局定义的 EMBEDDER)
            # 归一化后，欧氏距离的平方等于余弦距离，非常适合 DBSCAN
            vectors = EMBEDDER.encode(texts_to_embed, normalize_embeddings=True)
            
            # 3. DBSCAN 密度聚类
            # min_samples=2 表示至少2篇相关文献才能组成一个“技术簇”
            min_samples = 2 if len(records) < 30 else 3
            # 修改后 (建议初始设定为 0.75，相当于要求余弦相似度大于 0.71)
            clusterer = DBSCAN(eps=0.75, min_samples=min_samples, metric='euclidean')
            labels = clusterer.fit_predict(vectors)

            # 4. 结果分组过滤
            clusters_dict = defaultdict(list)
            noise_count = 0
            for idx, label in enumerate(labels):
                if label >= 0:  # DBSCAN 中 label 为 -1 表示孤立的噪声点
                    clusters_dict[label].append(records[idx])
                else:
                    noise_count += 1

            if not clusters_dict:
                print("⚠️ 聚类算法未发现明显密度簇，所有文献的话题过于发散，均被判定为孤立点。")
                return []

            print(f"✅ 聚类完成！共发现 {len(clusters_dict)} 个紧密的技术簇，剔除了 {noise_count} 条孤立数据。")
            
            # 👇 核心修改 调用大模型为每一个聚类簇提取技术名称
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
                        cluster_text_parts.append(f"标题 {title}")
                    if method:
                        cluster_text_parts.append(f"方法 {method}")
                        
                # 截取适量文本防止 Token 超限
                cluster_context = "\n".join(cluster_text_parts)[:2000]
                
                system_prompt = "你是一个科技情报专家。请根据提供的文献标题和方法，总结出一个精准、专业的核心技术名称（只需输出纯文本名称，控制在15个字以内，绝对不要使用加粗、不要使用冒号，不要任何解释性文字）。"
                
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"文献内容如下\n{cluster_context}"}
                    ]
                    response = model(messages)
                    cluster_name = response.content if hasattr(response, 'content') else str(response)
                    
                    # 清洗大模型输出，去除可能带有的多余符号
                    cluster_name = cluster_name.strip('`"\'。.,\n ')
                    
                    # 兜底机制 确保名称不为空
                    if not cluster_name:
                        cluster_name = f"技术簇_{cluster_id + 1}"
                        
                except Exception as e:
                    print(f"⚠️ 簇 {cluster_id + 1} 命名失败 {str(e)}")
                    cluster_name = f"技术簇_{cluster_id + 1}"

                formatted_clusters.append({
                    "cluster_label": cluster_name,
                    "papers": papers
                })
                
            print("✅ 智能命名完成！")
            return formatted_clusters
            
        except Exception as e:
            print(f"❌ 聚类过程发生底层错误 {str(e)}")
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

                return {
                    "meta": {"source": "arxiv", "query": query, "count": len(standardized_results)},
                    "results": standardized_results
                }

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
            "description": "每批并行处理的数据条数，默认为2",
            "nullable": True,
            "default": 2
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

    def forward(self, raw_data_list: list, target_type: str, batch_size: int = 2, max_workers: int = 3) -> List[Dict[str, Any]]:
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


class TechSignalEvaluatorTool(Tool):
    """
    综合技术信号评估工具。
    同时识别前沿技术和弱信号，满足任一设定的阈值即可保留，并通过 signal_types 字段区分。
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
        for author_info in authors_list:
            if not author_info:
                continue
            match = re.search(r'\((.*?)\)', author_info)
            if match:
                insts = match.group(1).split(',')
                for inst in insts:
                    institutions.add(inst.strip())
            else:
                parts = re.split(r'[,;]', author_info)
                if len(parts) > 1:
                    institutions.add(parts[-1].strip())
                else:
                    institutions.add(author_info.strip())
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

            # 特征统计变量初始化
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
                cluster_copy["signal_types"] = signal_types
                
                # 保留详细判定理由
                if is_frontier:
                    cluster_copy["frontier_conditions"] = frontier_conditions
                if is_weak:
                    cluster_copy["weak_signal_conditions"] = weak_conditions
                    
                # 合并所有命中的条件到一个兼容字段中，方便你后游的 tech_connotation_extractor 直接读取注入
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

class TechConnotationExtractorTool(Tool):
    
    name = "tech_connotation_extractor"
    description = "对过滤后的技术簇进行内涵提取 利用大模型阅读簇内论文 提取基本定义 痛点 原理 体系 指标和价值6个维度 并将这些字段原地注入到原聚类簇字典中返回"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "经过评估工具输出的包含 cluster_label 和 papers 等字段的聚类簇列表"
        }
    }
    
    output_type = "any"

    def __init__(self):
        super().__init__()

    def forward(self, clusters_data: list) -> list:
        if not isinstance(clusters_data, list):
            return [{"error": "输入必须是列表格式"}]
            
        if not clusters_data:
            print("输入的簇列表为空 无需提取")
            return []

        model = ModelProvider.get_model()
        updated_clusters = []
        
        print(f"正在调用大模型对 {len(clusters_data)} 个技术簇进行技术内涵提取")

        system_prompt = """
        你是一个顶级的科技情报分析专家 请基于输入的文献摘要与核心内容 进行深度的情报提取
        
        提取与生成规范
        1 强制中文 必须100%转化为专业流畅的学术中文
        2 纯文本无标记 生成的内容必须是纯文本段落 绝对禁止在文本内部使用加粗和冒号等标记
        3 必须严格输出以下 JSON 格式 绝对不要输出 Markdown 标记 不要输出任何解释性文字
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
            
            print(f"正在抽取 {cluster_label} 的技术内涵")
            
            combined_text = ""
            for paper in papers:
                title = paper.get("标题", "")
                abstract = paper.get("摘要", "")
                method = paper.get("核心方法/方案总结", "")
                combined_text += f"标题 {title}\n摘要 {abstract}\n方法 {method}\n\n"
            
            combined_text = combined_text[:8000]

            updated_cluster = cluster.copy()

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请总结以下文献内容\n{combined_text}"} 
                ]
                
                response = model(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                clean = content.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                
                try:
                    result_dict = json.loads(clean)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                    if match:
                        result_dict = json.loads(match.group(0))
                    else:
                        result_dict = {}
                
                expected_keys = ["基本定义", "拟解决科学问题/痛点", "核心原理", "技术体系", "技术指标", "作用价值"]
                for key in expected_keys:
                    updated_cluster[key] = result_dict.get(key, "模型未能按要求生成此字段")

            except Exception as e:
                print(f"抽取失败 {str(e)}")
                for key in ["基本定义", "拟解决科学问题/痛点", "核心原理", "技术体系", "技术指标", "作用价值"]:
                    updated_cluster[key] = f"抽取异常 {str(e)}"

            updated_clusters.append(updated_cluster)

        print("内涵提取完成 已将核心字段成功注入原数据结构")
        return updated_clusters
    
from smolagents import Tool
import json
import re
from utils.common_utils import ModelProvider

class TechEvolutionAnalyzerTool(Tool):
    """
    技术演进路径与重点技术态势分析工具。
    输入为包含 papers 的聚类簇列表。利用大模型抽取技术的主流路线、瓶颈、实现方案、最新动向、未来趋势及文本化的技术演化图，并追加到原数据结构中。
    """
    
    name = "tech_evolution_analyzer"
    description = "对技术簇进行全面的演进态势分析。利用大模型提取主流技术路线、难题瓶颈、实现方案、新进发展动向、未来发展趋势、技术演化图等6个维度，原地注入到原聚类簇字典中返回。"
    
    inputs = {
        "clusters_data": {
            "type": "array",
            "description": "包含 papers 字段的聚类簇列表数据。"
        }
    }
    
    output_type = "any"

    def __init__(self):
        super().__init__()

    def forward(self, clusters_data: list) -> list:
        if not isinstance(clusters_data, list):
            return [{"error": "输入必须是列表格式"}]
            
        if not clusters_data:
            print("⚠️ 输入的簇列表为空，无需分析。")
            return []

        model = ModelProvider.get_model()
        updated_clusters = []
        
        print(f"🧠 正在调用大模型对 {len(clusters_data)} 个技术簇进行【技术演进路径与重点态势分析】...")

        # 严格约束：根据用户偏好，禁止生成加粗和内部冒号
        system_prompt = """
        你是一个顶级的科技情报与技术战略分析专家。请基于输入的文献摘要与核心内容，进行技术演进态势的深度结构化提取。
        
        【⚠️ 提取与生成规范】（最高优先级）
        1. 强制中文：必须100%转化为专业流畅的学术与战略情报中文。
        2. 纯文本无标记：生成的内容必须是纯文本。绝对禁止在文本内部使用加粗（**）和冒号（：）等Markdown标记。
        3. 必须严格输出以下 JSON 格式（绝对不要输出任何解释性文字）：
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
            
            print(f"  > 正在分析 [{cluster_label}] 的演进态势...")
            
            # 拼接该簇内的文本
            combined_text = ""
            for paper in papers:
                title = paper.get("标题", "")
                abstract = paper.get("摘要", "")
                method = paper.get("核心方法/方案总结", "")
                # 拼接时不使用冒号，防止干扰大模型对纯文本指令的理解
                combined_text += f"标题 {title}\n摘要 {abstract}\n方法 {method}\n\n"
            
            # 防止文本过长爆Token，截取前8000个字符
            combined_text = combined_text[:8000]

            # 复制原有的聚类字典，准备在里面追加新字段
            updated_cluster = cluster.copy()

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请深度分析以下文献内容\n{combined_text}"} 
                ]
                
                response = model(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # 强健的 JSON 清洗提取
                clean = content.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                
                try:
                    result_dict = json.loads(clean)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                    if match:
                        result_dict = json.loads(match.group(0))
                    else:
                        result_dict = {}
                
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

        print("✅ 演进态势分析完成！已将所有扩展字段成功注入原数据结构。")
        return updated_clusters