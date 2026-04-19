import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import re
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from smolagents import Tool
# ... existing code ...
import ast
# 引入你的统一模型提供者
import utils.common_utils as common_utils
import fitz  # PyMuPDF，极速且能精准拿页码
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 全局单例加载模型，利用 Mac M 系列芯片的 MPS 加速（或 CPU）
# 推荐使用 BAAI 针对学术优化的轻量级模型
EMBEDDER = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu') 
RERANKER = CrossEncoder('BAAI/bge-reranker-base', device='cpu')


import os
import urllib.request
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import re

# 建议在类外部定义缓存，避免重复加载
PDF_CACHE = {}
WEB_CACHE = {}

# 【新增】全局文献元数据字典，用于最后生成标准的参考文献表
GLOBAL_META = {} 

class UniversalRAGTool(Tool):
    name = "tool_universal_rag"
    description = "全能RAG工具。能同时解析本地PDF和在线网页URL，提取与核心论点最相关的干货片段。"
    inputs = {
        "query": {"type": "string", "description": "核心论点"},
        "local_papers": {"type": "array", "description": "本地文献 [{'id':'...', 'local_path':'...'}]"},
        "online_urls": {"type": "array", "description": "在线URL列表 ['http...']"},
        "top_k": {"type": "integer", "description": "提取数量", "nullable": True}
    }
    output_type = "string"

    def forward(self, query: str, local_papers: list, online_urls: list, top_k: int = 5) -> str:
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      # 扩大到1200字，足以包含一个完整的复杂学术段落
            chunk_overlap=250,    # 增加重叠度，防止关键的“上下文承接”被切断
            separators=["\n\n", "。", "\n", ".", "；"] # 优先按双换行和句号切
        )
        print(f"[RAG] 启动混合检索 | Query: {query[:15]}... | PDF: {len(local_papers)}篇 | Web: {len(online_urls)}个")

        # 1. 解析本地 PDF 并汇入总池
        for paper in local_papers:
            pid, path = paper.get('id'), paper.get('local_path')
            if path and os.path.exists(path):
                # 【新增】提取 PDF 元数据存入 GLOBAL_META
                if pid not in GLOBAL_META:
                    try:
                        with fitz.open(path) as pdf:
                            meta = pdf.metadata
                            GLOBAL_META[pid] = {
                                "type": "paper",
                                "title": meta.get("title") or os.path.basename(path),
                                "author": meta.get("author") or "Unknown Author",
                                "year": meta.get("creationDate", "    ")[2:6] if meta.get("creationDate") else "N/A"
                            }
                    except Exception as e:
                        GLOBAL_META[pid] = {"type": "paper", "title": os.path.basename(path), "author": "Unknown", "year": "N/A"}

                if path not in PDF_CACHE:
                    chunks = []
                    try:
                        with fitz.open(path) as pdf:
                            for page_num, page in enumerate(pdf):
                                text = page.get_text("text").strip()
                                if len(text) > 50:
                                    for c in splitter.split_text(text):
                                        chunks.append({"source_id": pid, "page": f"第{page_num + 1}页", "text": c})
                        PDF_CACHE[path] = chunks
                    except Exception as e:
                        print(f"  ⚠️ PDF解析失败 {path}: {e}")
                all_chunks.extend(PDF_CACHE.get(path, []))

        # 2. 解析在线网页 (Jina Reader) 并汇入总池
        for url in online_urls:
            # 【新增】先用 URL 占位，防止提取失败没有数据
            if url not in GLOBAL_META:
                GLOBAL_META[url] = {"type": "web", "title": url, "url": url}

            if url not in WEB_CACHE:
                chunks = []
                try:
                    req = urllib.request.Request(f"https://r.jina.ai/{url}", headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=15) as response:
                        content = response.read().decode('utf-8')
                        
                        # 【新增】从 Jina 的 Markdown 返回中截取网页标题
                        for line in content.split('\n')[:15]:
                            if line.startswith("Title: "):
                                GLOBAL_META[url]["title"] = line.replace("Title: ", "").strip()
                                break

                        for c in splitter.split_text(content):
                            chunks.append({"source_id": url, "page": "网页正文", "text": c})
                    WEB_CACHE[url] = chunks
                except Exception as e:
                    print(f"  ⚠️ 网页提取失败 {url}: {e}")
            all_chunks.extend(WEB_CACHE.get(url, []))

        if not all_chunks: 
            return "未提取到可用文本，请直接基于常识撰写。"

        # 3. 混合池双路检索与重排 (逻辑完全保持你原本的设计)
        tokenized = [list(c["text"]) for c in all_chunks]
        bm25_scores = BM25Okapi(tokenized).get_scores(list(query))
        
        c_emb = EMBEDDER.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
        q_emb = EMBEDDER.encode(query, normalize_embeddings=True)
        vec_scores = np.dot(c_emb, q_emb)
        
        bn = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        vn = (vec_scores - np.min(vec_scores)) / (np.max(vec_scores) - np.min(vec_scores) + 1e-9)
        hybrid = 0.4 * bn + 0.6 * vn
        
        top_n_candidates = min(30, len(all_chunks))
        candidates = [all_chunks[i] for i in np.argsort(hybrid)[::-1][:top_n_candidates]]
        
        cross_inp = [[query, c["text"]] for c in candidates]
        rerank_scores = RERANKER.predict(cross_inp)
        
        final_k = min(top_k, len(candidates))
        final_chunks = [candidates[i] for i in np.argsort(rerank_scores)[::-1][:final_k]]
        
        # 4. 组装语料
        res = f"【检索查询】: {query}\n\n"
        for i, c in enumerate(final_chunks):
            # 【核心修改】强制给来源加上 `REF:` 前缀，这是为了防止大模型将它和普通的数字混淆
            res += f"--- 论据 {i+1} ---\n📑 来源标号: [REF:{c['source_id']}]\n📄 定位: {c['page']}\n📝 原文: {c['text']}\n\n"
        # 5. 组装语料，精准透出出处和【作者信息】
        res = f"【检索查询】: {query}\n\n"
        for i, c in enumerate(final_chunks):
            ref_id = c['source_id']
            meta = GLOBAL_META.get(ref_id, {})
            # 尝试提取作者和年份
            author = meta.get("author", "某研究团队")
            year = meta.get("year", "近年")
            
            res += f"--- 论据 {i+1} ---\n"
            res += f"🧑‍🔬 学者/机构: {author} ({year})\n"
            res += f"📑 来源标号: [REF:{ref_id}]\n"
            res += f"📄 定位: {c['page']}\n"
            res += f"📝 原文: {c['text']}\n\n"
            
        return res

class AcademicRAGTool(Tool):
    name = "fine_rag"
    description = "工业级学术RAG工具。通过混合检索(向量+BM25)和Cross-Encoder重排，从指定的本地PDF中精准提取支撑论点的文献片段，并附带精确页码。"
    inputs = {
        "query": {
            "type": "string",
            "description": "当前章节的核心论点或查询问题。"
        },
        "papers": {
            "type": "array",
            "description": "字典列表，格式为 [{'id': 'paper_1', 'local_path': '/.../xx.pdf'}]"
        },
        "top_k": {
            "type": "integer",
            "description": "最终返回的高浓度片段数量，默认为 5。",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, query: str, papers: list, top_k: int = 5) -> str:
        all_chunks = []
        
        # 1. 结构化解析与语义切块 (带缓存)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?"]
        )

        for paper in papers:
            pid = paper.get('id')
            path = paper.get('local_path')
            if not path or not os.path.exists(path):
                continue
                
            if path not in PDF_CACHE:
                doc_chunks = []
                try:
                    with fitz.open(path) as pdf:
                        for page_num, page in enumerate(pdf):
                            text = page.get_text("text").strip()
                            if len(text) < 50: continue # 过滤空白页
                            
                            # 切块并注入元数据
                            chunks = splitter.split_text(text)
                            for chunk in chunks:
                                doc_chunks.append({
                                    "paper_id": pid,
                                    "page": page_num + 1,
                                    "text": chunk
                                })
                    PDF_CACHE[path] = doc_chunks
                except Exception as e:
                    print(f"[RAG] 解析 {pid} 失败: {e}")
                    continue
                    
            all_chunks.extend(PDF_CACHE[path])

        if not all_chunks:
            return "未提取到任何有效文本。"

        # 2. 混合检索 (Hybrid Retrieval) - 召回 Top 20
        # 2.1 BM25 稀疏检索 (抓取关键词)
        tokenized_corpus = [list(c["text"]) for c in all_chunks] # 中文简单按字tokenize，英文需分词
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(list(query))
        
        # 2.2 向量检索 (抓取语义)
        corpus_embeddings = EMBEDDER.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
        query_embedding = EMBEDDER.encode(query, normalize_embeddings=True)
        vector_scores = np.dot(corpus_embeddings, query_embedding)
        
        # 2.3 归一化并融合分数 (BM25占40%，向量占60%)
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        vector_norm = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + 1e-9)
        hybrid_scores = 0.4 * bm25_norm + 0.6 * vector_norm
        
        # 获取混合检索的 Top 20 候选
        top_20_indices = np.argsort(hybrid_scores)[::-1][:20]
        candidates = [all_chunks[i] for i in top_20_indices]

        # 3. 交叉编码器重排 (Cross-Encoder Re-ranking) - 精选 Top K
        # 将 Query 和 候选文本拼成 Pair 进行打分，这是精度跃升的关键
        cross_inp = [[query, c["text"]] for c in candidates]
        rerank_scores = RERANKER.predict(cross_inp)
        
        # 按重排分数排序
        final_indices = np.argsort(rerank_scores)[::-1][:top_k]
        final_chunks = [candidates[i] for i in final_indices]

        # 4. 组装为高度结构化的学术 Context
        formatted_context = f"【检索查询】: {query}\n\n"
        for idx, chunk in enumerate(final_chunks):
            formatted_context += (
                f"--- 核心论据 {idx+1} ---\n"
                f"📑 来源标识: [{chunk['paper_id']}]\n"
                f"📄 定位页码: 第 {chunk['page']} 页\n"
                f"📝 原文片段: {chunk['text']}\n\n"
            )
            
        return formatted_context

# =====================================================================
# 工具 1：学术广度搜索工具 (The Foraging Tool)
# =====================================================================
class AcademicSearchTool(Tool):
    name = "tool_academic_search"
    description = "用于在 ArXiv 或 OpenAlex 上检索学术论文。输入英文关键词，返回论文基础信息的 JSON 列表。"
    inputs = {
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "英文关键词列表"
        },
        "engine": {
            "type": "string", 
            "description": "'arxiv' 或 'openalex'", 
            "nullable": True
        },
        "sort_by": {
            "type": "string",
            "description": "排序方式。可选 'date' 或 'citation'。强烈建议使用默认的 'date'",
            "nullable": True
        },
        "max_results_per_query": {
            "type": "integer", 
            "description": "每个检索词的最大返回文献数，默认 10", # <=== 补上这行
            "nullable": True
        }
    }
    output_type = "any"

    def _reconstruct_openalex_abstract(self, inverted_index: dict) -> str:
        if not inverted_index: return "No abstract available."
        try:
            max_idx = max(max(positions) for positions in inverted_index.values())
            words = [""] * (max_idx + 1)
            for word, positions in inverted_index.items():
                for pos in positions:
                    words[pos] = word
            return " ".join(words).strip()
        except Exception:
            return "Abstract parsing error."

    def forward(self, search_queries: list, engine: str = "arxiv", sort_by: str = "date", max_results_per_query: int = 10) -> list:
        engine = (engine or "arxiv").lower()
        sort_by = (sort_by or "date").lower()
        max_results = max_results_per_query or 10
        all_papers = {} 
        
        for query in search_queries:
            try:
                if engine == "openalex":
                    encoded_query = urllib.parse.quote(query.strip())
                    sort_param = "cited_by_count:desc" if sort_by == "citation" else "relevance_score:desc"
                    url = f"https://api.openalex.org/works?search={encoded_query}&sort={sort_param}&per-page={max_results}"
                    headers = {'User-Agent': 'mailto:open_deep_research@example.com'}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        for paper in data.get('results', []):
                            title = paper.get('title')
                            if not title or title in all_papers: continue

                            # 获取会议/期刊信息
                            # 或者从 primary_location 获取
                            primary_location = paper.get('primary_location') or {}
                            source = primary_location.get('source') or {}
                            venue_name = source.get('display_name', '')
                            venue_type = source.get('type', '')
                            
                            # 获取 OpenAlex 作者信息
                            authorships = paper.get('authorships', [])
                            authors = [a.get('author', {}).get('display_name', '') for a in authorships]
                            authors = [a for a in authors if a]
                            author_str = ", ".join(authors) if authors else "Unknown Author"
                            
                            oa_data = paper.get('open_access', {})
                            raw_pdf_url = oa_data.get('oa_url', '') if oa_data.get('is_oa') else ""

                            pdf_url = ""
                            if raw_pdf_url and isinstance(raw_pdf_url, str):
                                if not raw_pdf_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                                    pdf_url = raw_pdf_url

                            all_papers[title] = {
                                "id": paper.get('id', ''),
                                "title": title,
                                "year": str(paper.get('publication_year', 'Unknown')),
                                "author": author_str,  # <=== 存入作者信息
                                "abstract": self._reconstruct_openalex_abstract(paper.get('abstract_inverted_index')),
                                "pdf_url": pdf_url, 
                                "source_query": query,
                                "venue": venue_name,      # 新增：会议/期刊名称
                                "venue_type": venue_type, # 新增：journal 或 conference
                            }
                else: 
                    encoded_query = urllib.parse.quote(f'all:{query.strip()}')
                    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=15) as response:
                        root = ET.fromstring(response.read().decode('utf-8'))
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        for entry in root.findall('atom:entry', ns):
                            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                            if not title or title in all_papers: continue
                            
                            # 获取 ArXiv 作者信息
                            author_elements = entry.findall('atom:author/atom:name', ns)
                            authors = [a.text.strip() for a in author_elements if a.text]
                            author_str = ", ".join(authors) if authors else "Unknown Author"
                            
                            paper_id = entry.find('atom:id', ns).text.strip()
                            pdf_url = paper_id.replace('/abs/', '/pdf/') + ".pdf" 
                            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                            
                            all_papers[title] = {
                                "id": paper_id,
                                "title": title,
                                "year": entry.find('atom:published', ns).text.split('-')[0],
                                "author": author_str,  # <=== 存入作者信息
                                "abstract": abstract,
                                "pdf_url": pdf_url,
                                "source_query": query
                            }
            except Exception as e:
                print(f"[AcademicSearchTool] 检索词 '{query}' 发生异常: {e}")
        if "retrieved_papers" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["retrieved_papers"] = []
        GLOBAL_MEMORY["retrieved_papers"].extend(list(all_papers.values()))
        return list(all_papers.values())


# =====================================================================
# 工具 2：摘要洞察提纯工具 (The Map Tool / Extractor)
# =====================================================================
class InsightExtractorTool(Tool):
    name = "tool_insight_extractor"
    description = "将多篇学术论文的长摘要压缩并提取核心突破与痛点。使用多线程并发处理，有效降低后续生成大纲时的上下文压力。"
    inputs = {
        "raw_papers": {
            "type": "array",
            "items": {"type": "object"},
            "description": "包含 id, title, abstract 的字典列表"
        }
    }
    output_type = "any"

    def _extract_single_insight(self, paper: dict, model) -> dict:
        """使用传入的 smolagents 模型实例并发提纯单篇摘要"""
        abstract = paper.get('abstract', '')
        if len(abstract) < 50:
            paper['insight'] = "摘要过短，无明确洞察"
            return paper

        prompt = f"""请用极度精简的中文提取以下摘要的核心情报。
要求：严格按照“【核心突破】：...；【局限/痛点】：...”的格式输出，总字数不超过 60 字。
摘要内容：{abstract}"""

        # 组装 smolagents 所需的消息格式
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = model(messages)
            # 兼容不同模型的返回格式
            insight = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            insight = f"提炼失败: {e}"

        return {
            "id": paper.get('id'),
            "title": paper.get('title'),
            "year": paper.get('year'),
            "insight": insight.strip(),
            "pdf_url": paper.get('pdf_url'),
            "source_query": paper.get('source_query')
        }

    def forward(self, raw_papers: list) -> list:
        # 从全局 common_utils 获取模型实例
        model = common_utils.ModelProvider.get_model()
        compressed_papers = []
        
        print(f"[InsightExtractorTool] 开始并发提炼 {len(raw_papers)} 篇摘要...")
        # 保持多线程加速，smolagents 模型请求(底层 HTTP)一般线程安全
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_paper = {executor.submit(self._extract_single_insight, p, model): p for p in raw_papers}
            for future in as_completed(future_to_paper):
                try:
                    result = future.result()
                    compressed_papers.append(result)
                except Exception as e:
                    print(f"[InsightExtractorTool] 某篇处理异常: {e}")

        return compressed_papers


# =====================================================================
# 工具 3：语义聚类工具 (The Reduce Preparer)
# =====================================================================
class SemanticClusterTool(Tool):
    name = "tool_semantic_cluster"
    description = "将提纯后的文献情报进行主题聚类，将其划分为 4-7 个技术流派或核心议题，输出分类后的 JSON 数据。"
    inputs = {
        "compressed_papers": {
            "type": "array",
            "items": {"type": "object"},
            "description": "包含 insight 字段的文献字典列表"
        }
    }
    output_type = "any"

    def forward(self, compressed_papers: list) -> dict:
        # 从全局 common_utils 获取模型实例
        model = common_utils.ModelProvider.get_model()

        papers_str = "\n".join([f"ID: {p['id']} | 标题: {p['title']} | 洞察: {p['insight']}" for p in compressed_papers])
        
        prompt = f"""你是一个高级学术架构师。请将以下文献情报按技术流派、发展阶段或核心痛点，聚类为 4 到 7 个高度凝练的主题（Theme）。
要求返回合法的 JSON 格式，结构如下：
{{
  "clusters": [
    {{
      "theme": "主题名称",
      "summary": "该主题的核心趋势与矛盾 (50字以内)",
      "paper_ids": ["ID1", "ID2"]
    }}
  ]
}}

文献情报如下：
{papers_str}"""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = model(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 清洗大模型可能会附带的 Markdown 代码块标签 (防 JSON 解析崩溃)
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            elif clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
                
            cluster_dict = json.loads(clean_content.strip())
            
            # 将完整的文献元数据挂载回聚类树中
            paper_lookup = {p['id']: p for p in compressed_papers}
            for cluster in cluster_dict.get('clusters', []):
                cluster['papers'] = [paper_lookup[pid] for pid in cluster.get('paper_ids', []) if pid in paper_lookup]
            return cluster_dict
            
        except Exception as e:
            print(f"[SemanticClusterTool] 聚类/解析失败: {e}")
            return {"clusters": [], "error": str(e)}


# =====================================================================
# 工具 4：大纲基建下载器 (The Downloader Tool)
# =====================================================================
import os
import urllib.request
import urllib.error
import ssl
import time
import hashlib
from smolagents import Tool  # 请根据你的框架调整

class PaperDownloaderTool(Tool):
    name = "tool_paper_downloader"
    description = "批量下载PDF文献。支持传入任意复杂的嵌套大纲字典或列表，工具会自动递归提取其中所有的 PDF URL 并进行下载。"
    inputs = {
        "papers": {
            "type": "any",
            "description": "包含文献URL的任意数据结构（可以是提取好的列表，也可以是整个嵌套的大纲字典）。"
        },
        "save_dir": {
            "type": "string",
            "description": "保存PDF的本地目录路径。"
        }
    }
    output_type = "any"

    def forward(self, papers, save_dir: str) -> dict:
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        # --- 🚀 核心修复：递归提取任意结构中的合法 URL 🚀 ---
        urls_to_download = set()
        def extract_urls(node):
            if isinstance(node, dict):
                # 检查字典本身是否包含 url 字段
                for key in ["pdf_url", "url", "link"]:
                    val = node.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        urls_to_download.add(val.strip())
                # 继续深层遍历
                for k, v in node.items():
                    if k == "supporting_papers" and isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.startswith("http"):
                                urls_to_download.add(item.strip())
                            elif isinstance(item, dict):
                                extract_urls(item)
                    else:
                        extract_urls(v)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str) and item.startswith("http"):
                        urls_to_download.add(item.strip())
                    else:
                        extract_urls(item)

        extract_urls(papers)
        url_list = list(urls_to_download)
        
        print(f"[PaperDownloaderTool] 智能扫描完成，共发现 {len(url_list)} 篇待下载文献...")

        # 绕过严格的 SSL 证书校验
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # 浏览器伪装
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
        }

        for url in url_list:
            # 过滤明显不是 PDF 的假链接（图片等）
            if url.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
                print(f"  [-] 跳过非PDF链接: {url}")
                continue

            # 使用 URL 的哈希值或截取生成安全的 ID
            paper_id = "paper_" + hashlib.md5(url.encode()).hexdigest()[:8]
            safe_filename = paper_id + ".pdf"
            local_path = os.path.join(save_dir,"pdfs", safe_filename)

            success = False
            for attempt in range(2):
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                        content_type = response.headers.get('Content-Type', '').lower()
                        # 放宽校验：只要不是网页(text/html)就尝试保存
                        if 'text/html' in content_type:
                            print(f"  [-] 下载失败: 目标是网页防爬拦截 (Content-Type: {content_type})")
                            break
                            
                        with open(local_path, 'wb') as f:
                            f.write(response.read())
                            
                        print(f"  [+] 成功下载: {url}")
                        results[paper_id] = {
                            "status": "success", 
                            "local_path": local_path,
                            "pdf_url": url,
                            "id": paper_id
                        }
                        success = True
                        break 
                        
                except urllib.error.HTTPError as e:
                    print(f"  [-] 下载失败 HTTP {e.code}: {url}")
                    if e.code in [403, 404]: break 
                    time.sleep(2)
                except Exception as e:
                    print(f"  [-] 下载异常 {str(e)[:30]}: {url}")
                    time.sleep(2)
            
            if not success:
                results[paper_id] = {"status": "failed", "pdf_url": url}

        return results
class SaveFileTool(Tool):
    name = "save_file"
    description = "安全保存文件工具。支持穿透沙盒写文件，可选择覆盖写入或追加写入。"
    
    inputs = {
        "content": {"type": "any", "description": "要保存的内容"},
        "out_dir": {"type": "string", "description": "输出目录路径"},
        "file_name": {"type": "string", "description": "文件名（不含后缀）"},
        "out_type": {"type": "string", "description": "文件后缀类型，如 md 或 json"},
        "append": {
            "type": "boolean", 
            "description": "是否追加写入。True 为追加，False 为覆盖。默认为 False。", 
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, content, out_dir: str, file_name: str, out_type: str, append: bool = False) -> str:
        import os
        import json
        
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{file_name}.{out_type.lower()}")
        
        try:
            if out_type.lower() == "json":
                # JSON 处理：强制转为 Python 对象
                new_content = json.loads(content) if isinstance(content, str) else content
                
                if append and os.path.exists(path) and os.path.getsize(path) > 0:
                    # 智能追加模式：先读取原有内容
                    with open(path, "r", encoding="utf-8") as f:
                        try:
                            existing_content = json.load(f)
                        except json.JSONDecodeError:
                            existing_content = []
                            
                    # 结构合并逻辑
                    if isinstance(existing_content, list):
                        if isinstance(new_content, list):
                            existing_content.extend(new_content)
                        else:
                            existing_content.append(new_content)
                        final_content = existing_content
                    elif isinstance(existing_content, dict) and isinstance(new_content, dict):
                        existing_content.update(new_content)
                        final_content = existing_content
                    else:
                        # 结构不兼容时，强行组装成列表
                        final_content = [existing_content, new_content]
                else:
                    final_content = new_content

                # 写回 JSON (即使是追加，JSON也必须全量覆盖重写以保证格式合法)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(final_content, f, ensure_ascii=False, indent=2)
                    
            else:
                # 非 JSON 文件（如 txt, md），直接根据 append 决定文件读写模式
                mode = "a" if append else "w"
                with open(path, mode, encoding="utf-8") as f:
                    # 追加模式下，为了美观，如果文件非空则在新内容前加个换行符
                    if append and os.path.exists(path) and os.path.getsize(path) > 0:
                        f.write("\n")
                    f.write(str(content))
                    
            mode_str = "追加" if append else "覆盖"
            return f"✅ 已成功{mode_str}保存至 {path}"
            
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"

import copy
from smolagents import Tool  # 请根据你的实际框架修改导入路径

class BindCitationsTool(Tool):
    name = "tool_bind_citations"
    description = "将下载好的本地 PDF 路径精准挂载到大纲的对应章节中。输入原始大纲和下载结果，返回挂载了本地路径的最终大纲。若下载失败，将保留原始URL。"
    inputs = {
        "outline_data": {
            "type": "object",
            "description": "由 Outliner 生成的原始大纲字典，其中应包含 supporting_papers 列表。"
        },
        "download_results": {
            "type": "any",
            "description": "由 tool_paper_downloader 返回的字典或列表，包含每个文献的下载状态和 local_path。"
        }
    }
    output_type = "any"

    def forward(self, outline_data: dict, download_results) -> dict:
        # 深拷贝以防污染原始数据
        enriched_outline = copy.deepcopy(outline_data)
        
        # 1. 兼容 download_results 可能是列表或字典的情况
        download_dict = {}
        if isinstance(download_results, dict):
            download_dict = download_results
        elif isinstance(download_results, list):
            for i, item in enumerate(download_results):
                if isinstance(item, dict):
                    # 提取 id，若无则使用索引兜底
                    did = str(item.get("id", i))
                    download_dict[did] = item

        # 2. 定义递归函数，自动寻找树状结构中的所有 supporting_papers
        def traverse_and_bind(node):
            if isinstance(node, dict):
                # 优先递归子节点（避开 supporting_papers 内部的字符串）
                for key, value in node.items():
                    if key != "supporting_papers":
                        traverse_and_bind(value)

                # 处理当前节点的 supporting_papers
                if "supporting_papers" in node and isinstance(node["supporting_papers"], list):
                    bound_papers = []
                    for paper_ref in node["supporting_papers"]:
                        # 兼容处理：防重复挂载时 paper_ref 已经是字典的情况
                        if isinstance(paper_ref, dict):
                            paper_id = str(paper_ref.get("url", paper_ref.get("id", "")))
                        else:
                            paper_id = str(paper_ref).strip()
                            
                        if not paper_id: 
                            continue
                        
                        # 在下载结果中匹配对应的 PDF 路径
                        match = None
                        for pid, info in download_dict.items():
                            if paper_id == str(pid) or paper_id == str(info.get("pdf_url", "")):
                                match = info
                                break
                        
                        # 🚀 核心修复区：无论成败，都不允许丢弃文献 🚀
                        if match and match.get("status") == "success":
                            # 下载成功：挂载本地路径
                            bound_papers.append({
                                "id": match.get("id", paper_id),
                                "url": match.get("pdf_url", paper_id),
                                "local_path": match.get("local_path", ""),
                                "status": "local_success"
                            })
                        else:
                            # 下载失败或未找到：保留原汁原味的在线 URL，明确标记无本地文件
                            bound_papers.append({
                                "id": paper_id,
                                "url": paper_id,
                                "local_path": None,
                                "status": "online_only"
                            })
                            
                    # 替换原本的列表为精细化的字典列表
                    node["supporting_papers"] = bound_papers

            elif isinstance(node, list):
                for item in node:
                    traverse_and_bind(item)

        print("[BindCitationsTool] 正在执行大纲文献的精准递归挂载...")
        traverse_and_bind(enriched_outline)
        return enriched_outline



# =====================================================================
# 工具 1：结构化容错解析器 (Parser Tool)
# 替代原本容易崩溃的 ast.literal_eval，安全剥离 Markdown 包装并解析 JSON
# =====================================================================
class ParseJsonTool(Tool):
    name = "tool_parse_json"
    description = "安全解析大模型输出的字符串，自动去除Markdown代码块，提取合法的Python字典或列表。专门用于替代 ast.literal_eval 以避免语法报错。"
    inputs = {
        "text": {"type": "string", "description": "大模型返回的原始文本"}
    }
    output_type = "any"

    def forward(self, text: str):
        if not isinstance(text, str):
            return text
            
        # 清理常见的 Markdown 包装
        cleaned = re.sub(r'^\`\`\`(?:json|python)?\n', '', text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\`\`\`$', '', cleaned.strip(), flags=re.MULTILINE).strip()
        
        try:
            return json.loads(cleaned)
        except Exception:
            # 如果 JSON 失败，尝试用正则提取最外层的 {} 或 [] 并用 ast 解析
            match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if match:
                try:
                    return ast.literal_eval(match.group(1))
                except Exception as e:
                    return {"error": f"AST解析失败: {str(e)}", "raw": text}
        return {"error": "未发现有效的数据结构", "raw": text}



class MatchCitationsTool(Tool):
    name = "tool_match_citations"
    description = "将大纲中 supporting_papers 里的文献引用（标题、ID或简写URL）精准映射为 paper_pool 中的合法 pdf_url。这能防止模型产生幻觉链接。"
    inputs = {
        "outline_tree": {
            "type": "any",
            "description": "JSON格式的大纲字典或列表，包含 supporting_papers 字段。"
        },
        "paper_pool": {
            "type": "array",
            "description": "原始检索返回的文献列表，包含 title, id, pdf_url 等字段。"
        }
    }
    output_type = "any"

    def forward(self, outline_tree, paper_pool):
        # 1. 防御性解析
        if isinstance(outline_tree, str):
            try:
                clean_str = outline_tree.strip().strip('`').replace('json\n', '')
                outline_tree = ast.literal_eval(clean_str) if '[' in clean_str or '{' in clean_str else json.loads(clean_str)
            except Exception:
                print("[MatchCitationsTool] 大纲解析失败，原样返回。")
                return outline_tree

        if isinstance(paper_pool, str):
            try:
                paper_pool = ast.literal_eval(paper_pool)
            except Exception:
                pass

        if not isinstance(paper_pool, list):
            return outline_tree

        # 2. 构建多维度文献映射池
        paper_map = {}
        for paper in paper_pool:
            if not isinstance(paper, dict): continue
            pdf_url = paper.get("pdf_url")
            if not pdf_url: continue  # 没有下载链接的直接忽略
            
            # 以 URL, title, id 为键建立反向索引（全部转小写去空）
            paper_map[str(pdf_url).strip()] = pdf_url
            if paper.get("title"): paper_map[str(paper["title"]).strip().lower()] = pdf_url
            if paper.get("id"): paper_map[str(paper["id"]).strip().lower()] = pdf_url

        # 3. 递归遍历大纲，执行精准/模糊替换
        def traverse_and_match(node):
            if isinstance(node, dict):
                if "supporting_papers" in node and isinstance(node["supporting_papers"], list):
                    new_papers = []
                    for ref in node["supporting_papers"]:
                        ref_lower = str(ref).strip().lower()
                        
                        # 尝试精确匹配
                        if ref_lower in paper_map:
                            new_papers.append(paper_map[ref_lower])
                        else:
                            # 尝试模糊包含匹配 (防止大模型截断了链接或标题)
                            matched = False
                            for key, real_url in paper_map.items():
                                if len(key) > 10 and (key in ref_lower or ref_lower in key):
                                    new_papers.append(real_url)
                                    matched = True
                                    break
                            # 如果实在匹配不到，但本身是合法URL格式，则予以保留
                            if not matched and str(ref).startswith("http"):
                                new_papers.append(str(ref).strip())
                                
                    node["supporting_papers"] = list(set(new_papers)) # 去重
                
                # 继续遍历其他节点
                for k, v in node.items():
                    if k != "supporting_papers":
                        traverse_and_match(v)
            elif isinstance(node, list):
                for item in node:
                    traverse_and_match(item)

        traverse_and_match(outline_tree)
        return outline_tree
class FlattenOutlineTool(Tool):
    name = "tool_flatten_outline"
    description = "将复杂的树状大纲展平为线性的写作任务列表，提取每个小节的标题、核心论点和待检索的文献。"
    inputs = {
        "outline_data": {
            "type": "any",
            "description": "解析后的大纲JSON对象（支持列表或字典格式）"
        }
    }
    output_type = "any"

    def forward(self, outline_data) -> dict:
        tasks = []
        main_title = "深度研究学术报告"
        
        # 兼容字典或列表格式
        if isinstance(outline_data, dict):
            main_title = outline_data.get("level_1_title", outline_data.get("chapter_title", "深度研究学术报告"))
            nodes = outline_data.get("chapters", outline_data.get("sub_chapters", outline_data.get("level_1", [outline_data])))
        elif isinstance(outline_data, list):
            nodes = outline_data
        else:
            nodes = []

        def traverse(node, depth):
            if not isinstance(node, dict): return
            
            title = node.get("chapter_title") or node.get("section_title") or node.get("subsection_title") or ""
            
            # 如果是叶子节点（有 supporting_papers）
            if "supporting_papers" in node:
                papers = node.get("supporting_papers", [])
                local_papers = [{"id": p["id"], "local_path": p["local_path"]} for p in papers if isinstance(p, dict) and p.get("status") == "local_success"]
                online_urls = [p.get("url") for p in papers if isinstance(p, dict) and p.get("status") == "online_only"]
                
                tasks.append({
                    "type": "section",
                    "depth": depth,
                    "title": title,
                    "core_argument": node.get("core_argument", ""),
                    "local_papers": local_papers,
                    "online_urls": online_urls
                })
            else:
                # 纯标题/导语节点
                if title or node.get("core_argument"):
                    tasks.append({
                        "type": "heading", 
                        "depth": depth, 
                        "title": title, 
                        "core_argument": node.get("core_argument", "")
                    })
                
                # 递归向下
                for k in ["chapters", "sections", "sub_chapters", "subsections", "level_2"]:
                    if k in node and isinstance(node[k], list):
                        for child in node[k]:
                            traverse(child, depth + 1)
        
        for n in nodes:
            traverse(n, 1)
            
        return {"main_title": main_title, "tasks": tasks}
class ReportAssemblerTool(Tool):
    name = "tool_assemble_report"
    description = "底层全自动循环执行器。传入展平后的任务列表，它会在后台自动调用 RAG 和写作智能体，完成全篇万字报告的组装与逻辑润色。"
    inputs = {
        "main_title": {"type": "string", "description": "文章总标题"},
        "tasks": {"type": "array", "description": "由 tool_flatten_outline 返回的任务列表"}
    }
    output_type = "string"

    def __init__(self, writer_agent, rag_tool):
        super().__init__()
        self.writer_agent = writer_agent
        self.rag_tool = rag_tool

    def forward(self, main_title: str, tasks: list) -> str:
        report_lines = [f"# {main_title}\n\n"]
        print(f"[ReportAssembler] 引擎启动，开始自动组装与润色 {len(tasks)} 个节点...")
        GLOBAL_MEMORY["report_list"] = []
        
        # 新增：用于在循环中追踪上一段落的内容
        previous_content = "" 

        for idx, task in enumerate(tasks):
            depth = task.get("depth", 1)
            title = task.get("title", "")
            node_type = task.get("type", "heading")
            core_arg = task.get("core_argument", "")

            # 打印标题
            if title:
                report_lines.append(f"{'#' * (depth+1)} {title}\n\n")

            if node_type == "heading":
                if core_arg:
                    report_lines.append(f"> *{core_arg}*\n\n")
            
            elif node_type == "section":
                # 1. 呼叫 RAG
                local_papers = task.get("local_papers", [])
                online_urls = task.get("online_urls", [])
                
                if local_papers or online_urls:
                    print(f"  [{idx}/{len(tasks)}] 🔍 RAG 正在提取 {title} 的语料...")
                    enhanced_query = f"{core_arg} 具体方法 实验数据 提升指标 性能对比 百分比"
                    rag_context = self.rag_tool.forward(query=enhanced_query, local_papers=local_papers, online_urls=online_urls, top_k=8)
                else:
                    rag_context = "无"

                # 2. 呼叫写作大模型生成初稿
                print(f"  [{idx}/{len(tasks)}] ✍️ 正在撰写初稿...")
                
                prompt = (
                    f"【待撰写章节】: {title}\n"
                    f"【核心论点】: {core_arg}\n"
                    f"【RAG 精准提取语料】:\n{rag_context}\n"
                    f"【写作强制红线】:\n"
                    f"1. 必须使用提供的 [REF:xxx] 格式进行引用（例如 [REF:paper_123] 或 [REF:https://...]）。\n"
                    f"2. 绝对禁止抄袭语料原文中自带的数字标号（如 [63]、[120]等）和生成参考文献表！\n"
                    f"3. 核心指令：拒绝空洞论述！你必须像写顶会论文的 Related Work 一样，精确提取语料中的数据进行论证。行文必须包含：【学者/机构名】、【提出的具体方法/模型名称】、【具体的实验参数或对比指标】、【提升的准确数值或百分比】。\n"
                    f"4. 绝不要凭空捏造数据，如果没有数据，宁可深挖方法的原理细节。"
                )
                
                try:
                    res = self.writer_agent.run(prompt)
                    draft_content = getattr(res, 'content', str(res))
                except Exception as e:
                    draft_content = f"撰写失败: {str(e)}"

                # 3. 呼叫大模型进行逻辑润色与引用校验
                print(f"  [{idx}/{len(tasks)}] 🔄 正在执行逻辑衔接与引用校验...")
                polish_prompt = f"""你是一名严谨的学术审校专家。请对下面的初稿进行润色与事实校验。
【上下文信息】
上一段落内容：{previous_content if previous_content else "（无）"}
本段核心论点：{core_arg}

【待润色初稿】
{draft_content}

【必须严格执行的润色指令】
1. 逻辑重构：平滑衔接上一段。绝不使用生硬的“机器味”过渡语。
2. 引用纠错：确保文中只出现 [REF:xxx] 格式的引用。彻底删除大模型照抄的无关标号（如 [63]）或自行生成的参考列表。
3. 保留硬核数据：绝对不能在润色时把初稿中提到的作者名、方法名、实验数值和百分比等关键信息删掉。必须保留学术浓度！
4. 格式锁定：直接输出纯正文文本，保留原始的 Markdown 格式，绝不输出额外的解释说明。
"""
                try:
                    # 直接调用底层的 LLM 执行润色（避免启动完整 Agent 导致额外的工具规划开销）
                    # 假设 self.writer_agent.model 是可以直接对话的模型实例
                    messages = [{"role": "user", "content": polish_prompt}]
                    polish_res = self.writer_agent.model(messages)
                    final_content = getattr(polish_res, 'content', str(polish_res)).strip()
                    
                    # 容错：如果大模型抽风加上了 ```markdown 等代码块，做一次清理
                    if final_content.startswith("```"):
                        final_content = final_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
                        
                except Exception as e:
                    print(f"  ⚠️ 润色失败，回退至初稿: {e}")
                    final_content = draft_content

                # 更新 previous_content 供下一个章节使用
                previous_content = final_content

                # 4. 落盘组装
                report_lines.append(final_content + "\n\n")
                GLOBAL_MEMORY["report_list"].append(
                    {
                        "idx": idx, 
                        "title": title, 
                        "content": final_content, 
                        "core_arg": core_arg, 
                        "rag_context": rag_context, 
                        "local_papers": local_papers, 
                        "online_urls": online_urls
                     }) 
                print(f"  ✅ 节点完成！")

        GLOBAL_MEMORY["final_report"] = "".join(report_lines)
        return "".join(report_lines)
    
from smolagents import Tool

# 物理层面上的全局共享内存
GLOBAL_MEMORY = {}

class SetVariableTool(Tool):
    name = "tool_set_var"
    description = "将复杂数据存入全局共享内存。"
    inputs = {
        "key": {"type": "string", "description": "变量名"},
        "value": {"type": "any", "description": "任何格式的数据"}
    }
    output_type = "string"

    def forward(self, key: str, value) -> str:
        GLOBAL_MEMORY[key] = value
        return f"✅ 数据已成功存入共享内存，键名: {key}"

class GetVariableTool(Tool):
    name = "tool_get_var"
    description = "从全局共享内存中读取复杂数据对象。"
    inputs = {
        "key": {"type": "string", "description": "变量名"}
    }
    output_type = "any" # 重点：直接返回 Python 对象，不转字符串

    def forward(self, key: str):
        if key not in GLOBAL_MEMORY:
            return f"❌ 错误：内存中不存在键名 {key}"
        return GLOBAL_MEMORY[key]
import re

class GenerateBibliographyTool(Tool):
    name = "tool_generate_bibliography"
    description = "扫描正文中的 [paper_xx] 或 [http..] 标签，将其替换为 [1][2] 格式，并在文末生成标准的学术参考文献列表。"
    inputs = {
        "report_content": {"type": "string", "description": "全文Markdown"},
        "tasks_data": {"type": "array", "description": "展平后的tasks列表(需包含文献的 title/author 等元数据)"}
    }
    output_type = "string"

    def forward(self, report_content: str, tasks_data: list) -> str:
        # 1. 构建全局文献元数据字典 (提取作者、标题、年份)
        meta_lookup = {}
        
        for task in tasks_data:
            # 兼容你的数据结构：不管是 local_papers 还是 papers_meta
            papers = task.get("papers_meta", []) or task.get("local_papers", [])
            for p in papers:
                pid = p.get("id", "")
                meta_lookup[pid] = {
                    "type": "paper",
                    "title": p.get("title", pid),           # 如果没有标题，退化为显示 ID
                    "author": p.get("author", "未知作者"),
                    "year": p.get("year", "N/A"),
                    "url": p.get("local_path", "")
                }
            
            # 记录在线网页的元数据
            for url in task.get("online_urls", []):
                meta_lookup[url] = {
                    "type": "web",
                    "title": url,  # 网页默认标题，如果外部有提取到可在此覆盖
                    "url": url
                }

        # 2. 正则查找文中所有的方括号引用
        citations = re.findall(r'\[([^\]]+)\]', report_content)
        
        unique_refs = []
        ref_mapping = {}  # 记录 原文本(包含可能的REF:前缀) -> 序号

        for cite in citations:
            # 处理如 [paper_1, http...] 这种逗号分隔的多个引用
            sub_cites = [c.strip() for c in cite.split(",")]
            for sc in sub_cites:
                # 过滤掉非文献引用（如 [注: 无本地文献]，保留 paper_ 和 http 开头的）
                clean_sc = sc.replace("REF:", "") # 兼容如果前面加了REF:的情况
                if clean_sc.startswith("paper_") or clean_sc.startswith("http"):
                    if clean_sc not in unique_refs:
                        unique_refs.append(clean_sc)
                    # 将原始字符串映射到最终的数字序号
                    ref_mapping[sc] = unique_refs.index(clean_sc) + 1

        # 3. 替换正文中的引用
        def replace_func(match):
            cite_str = match.group(1)
            sub_cites = [c.strip() for c in cite_str.split(",")]
            nums = []
            
            # 只有当括号里的内容全是我们识别到的文献时，才做替换
            for sc in sub_cites:
                if sc in ref_mapping:
                    nums.append(str(ref_mapping[sc]))
            
            # 如果成功映射，返回 [1,2]；否则原样返回保留原来的文本（例如 [1.1 核心瓶颈] 这种正常文本）
            if nums: 
                return f"[{','.join(nums)}]"
            return match.group(0)

        formatted_content = re.sub(r'\[([^\]]+)\]', replace_func, report_content)

        # 4. 追加标准格式的参考文献列表
        formatted_content += "\n\n---\n\n## 参考文献\n\n"
        for i, ref_id in enumerate(unique_refs):
            num_label = i + 1
            meta = meta_lookup.get(ref_id, {})
            
            # 判断是网页还是本地学术论文，输出不同格式
            if ref_id.startswith("http"):
                # 网页格式: [1] 标题. 取自: URL
                title = meta.get("title", ref_id)
                formatted_content += f"[{num_label}] {title}. 取自: {ref_id}\n"
            else:
                # 论文格式: [2] 作者. *论文标题*. 年份.
                author = meta.get("author", "未知作者")
                title = meta.get("title", ref_id)
                year = meta.get("year", "N/A")
                formatted_content += f"[{num_label}] {author}. *{title}*. {year}.\n"

        return formatted_content