import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import re
import ast
import copy
import time
import ssl
import hashlib
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import fitz  # PyMuPDF

import pypandoc
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool

# 引入你的统一模型提供者
import utils.common_utils as common_utils
# 全局单例加载模型，利用 Mac M 系列芯片的 MPS 加速（或 CPU）
# 推荐使用 BAAI 针对学术优化的轻量级模型
# 1. 动态获取当前脚本 (custom_tools.py) 的绝对路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 往上两级推算到 open_deep_research 的父目录，然后拼上 models 文件夹
project_root = os.path.dirname(os.path.dirname(current_script_dir))
models_dir = os.path.join(project_root, "models")

# 3. 动态拼装模型的绝对物理路径
embedder_path = os.path.join(models_dir, "bge-small-zh-v1.5")
reranker_path = os.path.join(models_dir, "bge-reranker-base")
# 4. 加载模型（如果本地路径不存在，可以留个优雅的报错提示）
if not os.path.exists(embedder_path) or not os.path.exists(reranker_path):
    raise FileNotFoundError(f"❌ 找不到本地模型！请确保把模型下载并解压到了: {models_dir}")

EMBEDDER = SentenceTransformer(embedder_path, device='cpu') 
RERANKER = CrossEncoder(reranker_path, device='cpu')

# 建议在类外部定义缓存，避免重复加载
PDF_CACHE = {}
WEB_CACHE = {}

# 物理层面上的全局共享内存
GLOBAL_MEMORY = {}
GLOBAL_MEMORY["PAPER_DB"] = {}  # 论文数据库，key为论文ID或URL，value为元信息字典（如标题、作者、年份等）
GLOBAL_MEMORY["retrieved_papers"] = []  # 记录已经检索过的论文元信息，避免重复调用 OpenAlex API
GLOBAL_MEMORY["report_list"] = []  # 存储生成报告的不同版本，便于后续分析和回溯
GLOBAL_MEMORY["final_report"] = ""  # 存储最终定稿的报告内容，供后续工具调用（如生成参考文献列表）
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
        import ssl
        import json
        import urllib.request
        import re
        import numpy as np
        import os
        import csv
        
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      
            chunk_overlap=250,    
            separators=["\n\n", "。", "\n", ".", "；"] 
        )
        
        # 1. Query 净化：剥离大模型强加的写作用指令词，提取纯净的学术语义用于检索
        clean_query = query
        stop_words = ["具体方法", "实验数据", "提升指标", "性能对比", "百分比", "具体数值", "详细阐述"]
        for word in stop_words:
            clean_query = clean_query.replace(word, "")
        clean_query = re.sub(r'[^\w\u4e00-\u9fa5a-zA-Z0-9]+', ' ', clean_query).strip()

        print(f"[RAG] 启动混合检索 | 净化Query: {clean_query[:25]}... | PDF: {len(local_papers)}篇 | Web: {len(online_urls)}个")

        # 2. 解析本地 PDF 并汇入总池 (🚀 注入 1pdf2md 视觉解析逻辑)
        for paper in local_papers:
            pid, path = paper.get('id'), paper.get('local_path')
            if path and os.path.exists(path):
                if pid not in GLOBAL_MEMORY.get("PAPER_DB", {}):
                    GLOBAL_MEMORY.setdefault("PAPER_DB", {})
                    try:
                        with fitz.open(path) as pdf:
                            meta = pdf.metadata
                            GLOBAL_MEMORY["PAPER_DB"][pid] = {
                                "type": "paper",
                                "title": meta.get("title") or os.path.basename(path),
                                "author": meta.get("author") or "Unknown Author",
                                "year": meta.get("creationDate", "    ")[2:6] if meta.get("creationDate") else "N/A"
                            }
                    except Exception:
                        GLOBAL_MEMORY["PAPER_DB"][pid] = {"type": "paper", "title": os.path.basename(path), "author": "Unknown", "year": "N/A"}

                if path not in PDF_CACHE:
                    chunks = []
                    
                    # 🚀 尝试读取 1pdf2md 的结构化 CSV
                    base_dir = os.path.dirname(os.path.dirname(path))
                    file_name = os.path.splitext(os.path.basename(path))[0]
                    csv_path = os.path.join(base_dir, "structured_csv", file_name, f"{file_name}.csv")
                    
                    if os.path.exists(csv_path):
                        try:
                            with open(csv_path, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    content = row.get('内容', '').strip()
                                    img_info = row.get('所需图片及其地址', '').strip()
                                    if len(content) > 10:
                                        for c in splitter.split_text(content):
                                            chunks.append({
                                                "source_id": pid, 
                                                "page": f"章节: {row.get('标题', '')}", 
                                                "text": c,
                                                "images": img_info if img_info and img_info != '无' else None
                                            })
                            print(f"  [+] 命中视觉增强数据: {csv_path}")
                        except Exception as e:
                            print(f"  ⚠️ CSV读取异常 {csv_path}: {e}")

                    # 🚀 降级方案：如果没有 CSV，走原有的 fitz 纯文本提取
                    if not chunks:
                        try:
                            with fitz.open(path) as pdf:
                                for page_num, page in enumerate(pdf):
                                    text = page.get_text("text").strip()
                                    if len(text) > 50:
                                        for c in splitter.split_text(text):
                                            chunks.append({
                                                "source_id": pid, 
                                                "page": f"第{page_num + 1}页", 
                                                "text": c, 
                                                "images": None
                                            })
                        except Exception as e:
                            print(f"  ⚠️ PDF解析失败 {path}: {e}")
                            
                    PDF_CACHE[path] = chunks
                    
                all_chunks.extend(PDF_CACHE.get(path, []))

        # 3. 解析在线网页并汇入总池 (包含 OpenAlex API 拦截和 SSL 绕过)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        memory_lookup = {}
        for p in GLOBAL_MEMORY.get("retrieved_papers", []):
            if p.get("id"): memory_lookup[p["id"]] = p

        for url in online_urls:
            if url not in GLOBAL_MEMORY.get("PAPER_DB", {}):
                GLOBAL_MEMORY.setdefault("PAPER_DB", {})
                GLOBAL_MEMORY["PAPER_DB"][url] = {"type": "web", "title": url, "url": url}

            if url not in WEB_CACHE:
                chunks = []
                is_processed = False
                
                oa_match = re.search(r'(W\d{8,})', url)
                if oa_match:
                    work_id = oa_match.group(1)
                    matched_paper = memory_lookup.get(work_id)
                    
                    if matched_paper and matched_paper.get("abstract") and matched_paper.get("abstract") != "No abstract available.":
                        title = matched_paper.get("title", url)
                        GLOBAL_MEMORY["PAPER_DB"][url].update({
                            "title": title,
                            "author": matched_paper.get("author", "未知学者"),
                            "year": str(matched_paper.get("year", "N/A"))
                        })
                        text_content = f"【论文标题】: {title}\n【完整摘要】: {matched_paper['abstract']}"
                        for c in splitter.split_text(text_content):
                            chunks.append({"source_id": url, "page": "文献摘要(内存)", "text": c})
                        is_processed = True
                        
                    if not is_processed:
                        try:
                            req = urllib.request.Request(f"https://api.openalex.org/works/{work_id}", headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=10) as response:
                                data = json.loads(response.read().decode('utf-8'))
                                
                                inv_idx = data.get('abstract_inverted_index')
                                abstract = "未提供摘要内容。"
                                if inv_idx:
                                    max_idx = max(max(positions) for positions in inv_idx.values())
                                    words = [""] * (max_idx + 1)
                                    for word, positions in inv_idx.items():
                                        for pos in positions: words[pos] = word
                                    abstract = " ".join(words).strip()
                                
                                title = data.get('title', url)
                                authorships = data.get('authorships', [])
                                author = authorships[0].get('author', {}).get('display_name', '未知学者') if authorships else '未知学者'
                                year = str(data.get('publication_year', 'N/A'))
                                
                                GLOBAL_MEMORY["PAPER_DB"][url].update({"title": title, "author": author, "year": year})
                                
                                text_content = f"【论文标题】: {title}\n【完整摘要】: {abstract}"
                                for c in splitter.split_text(text_content):
                                    chunks.append({"source_id": url, "page": "官方摘要(API直连)", "text": c})
                                is_processed = True
                        except Exception as e:
                            pass

                if not is_processed:
                    try:
                        req = urllib.request.Request(f"https://r.jina.ai/{url}", headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                            content = response.read().decode('utf-8')
                            for line in content.split('\n')[:15]:
                                if line.startswith("Title: "):
                                    GLOBAL_MEMORY["PAPER_DB"][url]["title"] = line.replace("Title: ", "").strip()
                                    break
                            for c in splitter.split_text(content):
                                chunks.append({"source_id": url, "page": "网页正文", "text": c})
                    except Exception as e:
                        pass
                        
                WEB_CACHE[url] = chunks
            
            all_chunks.extend(WEB_CACHE.get(url, []))

        if not all_chunks: 
            return "未提取到可用文本，请直接基于常识撰写。"

        # 4. 混合池双路检索与重排
        tokenized_corpus = [list(c["text"]) for c in all_chunks]
        tokenized_query = list(clean_query.replace(" ", ""))
        bm25_scores = BM25Okapi(tokenized_corpus).get_scores(tokenized_query)
        
        c_emb = EMBEDDER.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
        q_emb = EMBEDDER.encode(clean_query, normalize_embeddings=True)
        vec_scores = np.dot(c_emb, q_emb)
        
        bn = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        vn = (vec_scores - np.min(vec_scores)) / (np.max(vec_scores) - np.min(vec_scores) + 1e-9)
        hybrid = 0.4 * bn + 0.6 * vn
        
        top_n_candidates = min(30, len(all_chunks))
        candidates = [all_chunks[i] for i in np.argsort(hybrid)[::-1][:top_n_candidates]]
        
        cross_inp = [[clean_query, c["text"]] for c in candidates]
        rerank_scores = RERANKER.predict(cross_inp)
        
        # 5. 核心打散机制：强制同源去重
        sorted_indices = np.argsort(rerank_scores)[::-1]
        final_chunks = []
        source_counts = {}
        skipped_chunks = []
        
        # 强控参数：同一篇论文最多只能抽取 2 个片段
        max_chunks_per_source = 2 
        
        for idx in sorted_indices:
            c = candidates[idx]
            sid = c["source_id"]
            
            if source_counts.get(sid, 0) < max_chunks_per_source:
                final_chunks.append(c)
                source_counts[sid] = source_counts.get(sid, 0) + 1
            else:
                skipped_chunks.append(c)
                
            if len(final_chunks) >= top_k:
                break
                
        # 兜底机制：如果检索到的不同文献总数太少（导致没凑齐 top_k）
        # 则从刚刚被跳过的高分片段中补齐，保证大模型有足够的文字量
        if len(final_chunks) < top_k:
            needed = top_k - len(final_chunks)
            final_chunks.extend(skipped_chunks[:needed])
        
        # 6. 组装最终喂给大模型的语料 (🚀 注入图片/公式路径输出逻辑)
        res = f"【检索查询】: {query}\n\n" 
        for i, c in enumerate(final_chunks):
            ref_id = c['source_id']
            meta = GLOBAL_MEMORY.get("PAPER_DB", {}).get(ref_id, {})
            author = meta.get("author", "某研究团队")
            year = meta.get("year", "近年")
            
            res += f"--- 论据 {i+1} ---\n"
            res += f"🧑‍🔬 学者/机构: {author} ({year})\n"
            res += f"📑 来源标号: [REF:{ref_id}]\n"
            res += f"📄 定位: {c['page']}\n"
            res += f"📝 原文: {c['text']}\n"
            
            # 🚀 若存在视觉证据，则向大模型暴露物理路径
            if c.get('images'):
                res += f"🖼️ 视觉证据(配图/公式)信息:\n{c['images']}\n"
            else:
                res += "\n"
            
        # 打印打散后的来源统计
        print(f"  [+] ♻️ 打散完成，共选取 {len(final_chunks)} 个片段，来自 {len(source_counts)} 篇不同文献。")
            
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
    description = "用于在 ArXiv 或 OpenAlex 上检索学术论文。输入英文关键词，返回论文基础信息的 JSON 列表。会自动将检索结果双向同步到全局内存 `retrieved_papers` (列表) 和 `PAPER_DB` (字典字典) 中。"
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
            "description": "每个检索词的最大返回文献数，默认 10", 
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

    def forward(self, search_queries: list, engine: str = "openalex", sort_by: str = "date", max_results_per_query: int = 10) -> list:
        import time
        import urllib.error
        
        engine = (engine or "openalex").lower()
        sort_by = (sort_by or "date").lower()
        max_results = max_results_per_query or 10
        all_papers = {} 
        
        for query in search_queries:
            try:
                if engine == "openalex":
                    encoded_query = urllib.parse.quote(query.strip())
                    sort_param = "cited_by_count:desc" if sort_by == "citation" else "relevance_score:desc"
                    url = f"https://api.openalex.org/works?search={encoded_query}&sort={sort_param}&per-page={max_results}"
                    headers = {'User-Agent': 'mailto:wangchao@example.com'}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        for paper in data.get('results', []):
                            title = paper.get('title')
                            if not title or title in all_papers: continue

                            # 获取会议/期刊信息
                            primary_location = paper.get('primary_location') or {}
                            source = primary_location.get('source') or {}
                            venue_name = source.get('display_name', '')
                            venue_type = source.get('type', '')
                            
                            # 获取 OpenAlex 作者信息
                            authorships = paper.get('authorships', [])
                            authors = [a.get('author', {}).get('display_name', '') for a in authorships]
                            authors = [a for a in authors if a]
                            author_str = ", ".join(authors) if authors else "Unknown Author"

                            # 前三个作者，后面的用 et al.
                            author_display = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown Author"
                            
                            year = str(paper.get('publication_year', 'Unknown'))
                            oa_id = paper.get('id', '')
                            
                            # 组装 IEEE 风格引用
                            std_cite = f'{author_display}. "{title}."'
                            if venue_name: std_cite += f' *{venue_name}*,'
                            std_cite += f' {year}. 取自: {oa_id}'
                            
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
                                "author": author_str,
                                "abstract": self._reconstruct_openalex_abstract(paper.get('abstract_inverted_index')),
                                "pdf_url": pdf_url, 
                                "source_query": query,
                                "venue": venue_name,
                                "venue_type": venue_type,
                                "standard_citation": std_cite,
                            }
                else: 
                    # ==========================================
                    # 🚀 ArXiv 专属抗压优化方案
                    # ==========================================
                    
                    # 1. 净化检索词：ArXiv 原生检索引擎处理括号和布尔符极易 502，做降级打平处理
                    safe_query = query.replace(" AND ", " ").replace(" OR ", " ").replace("(", "").replace(")", "")
                    encoded_query = urllib.parse.quote(f'all:{safe_query.strip()}')
                    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
                    
                    # 2. 加入重试机制
                    max_retries = 3
                    success = False
                    for attempt in range(max_retries):
                        try:
                            # 3. 规范 User-Agent（留下真实邮箱标识，ArXiv 官方对此类请求更宽容，不易拉黑）
                            headers = {'User-Agent': 'open_deep_research_agent/1.0 (mailto:wangchao@example.com)'}
                            req = urllib.request.Request(url, headers=headers)
                            
                            with urllib.request.urlopen(req, timeout=15) as response:
                                root = ET.fromstring(response.read().decode('utf-8'))
                                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                                for entry in root.findall('atom:entry', ns):
                                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                                    if not title or title in all_papers: continue
                                    
                                    # 🚀 提取作者并拼装 standard_citation
                                    author_elements = entry.findall('atom:author/atom:name', ns)
                                    authors = [a.text.strip() for a in author_elements if a.text]
                                    author_display = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown Author"
                                    
                                    year = entry.find('atom:published', ns).text.split('-')[0]
                                    paper_id = entry.find('atom:id', ns).text.strip()
                                    pdf_url = paper_id.replace('/abs/', '/pdf/') + ".pdf" 
                                    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                                    
                                    # 组装 IEEE 风格引用
                                    std_cite = f'{author_display}. "{title}." *arXiv preprint*, {year}. 取自: {paper_id}'
                                    
                                    all_papers[title] = {
                                        "id": paper_id,
                                        "title": title,
                                        "year": year,
                                        "author": author_display,
                                        "abstract": abstract,
                                        "pdf_url": pdf_url,
                                        "source_query": query,
                                        "standard_citation": std_cite # <=== 新增入库标准格式
                                    }
                            success = True
                            break # 成功则跳出重试循环
                            
                        except urllib.error.HTTPError as e:
                            print(f"  [ArXiv API] 第 {attempt+1} 次请求失败: HTTP {e.code} ({query[:20]}...)")
                            time.sleep(4) # 遇到 HTTP 错误多等一会
                        except Exception as e:
                            print(f"  [ArXiv API] 第 {attempt+1} 次请求发生异常: {e}")
                            time.sleep(2)
                            
                    if not success:
                        print(f"  ⚠️ 检索词 '{query[:30]}...' 连续 {max_retries} 次获取失败，已跳过。")
                        
                    # 4. 强制频控排队：不论成功失败，请求下一个词之前强制休眠 3.5 秒
                    time.sleep(3.5)

            except Exception as e:
                print(f"[AcademicSearchTool] 检索词 '{query}' 发生全局异常: {e}")

        # ====================================================================
        # 🚀 核心架构升级：双路数据入库
        # ====================================================================
        
        # 1. 存入/更新 PAPER_DB (全局主键文献库 - 供下载器、RAG和参考文献生成使用)
        if "PAPER_DB" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["PAPER_DB"] = {}
            
        for paper_data in all_papers.values():
            paper_id = paper_data.get("id")
            if paper_id:
                # 存入字典，以 O(1) 极速查询
                GLOBAL_MEMORY["PAPER_DB"][paper_id] = paper_data

        # 2. 追加至 retrieved_papers (流水线列表 - 供提纯智能体做并发摘要使用)
        if "retrieved_papers" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["retrieved_papers"] = []
            
        # 提取当前列表中已有的文献 ID，转为集合(Set)实现 O(1) 极速去重查询
        existing_ids = {p.get("id") for p in GLOBAL_MEMORY["retrieved_papers"] if p.get("id")}
        
        # 只把没出现过的新文献加进去
        for paper in all_papers.values():
            pid = paper.get("id")
            if pid and pid not in existing_ids:
                GLOBAL_MEMORY["retrieved_papers"].append(paper)
        
        print(f"[AcademicSearchTool] 入库完毕！当前 retrieved_papers 累积总数: {len(GLOBAL_MEMORY['retrieved_papers'])} | PAPER_DB 词条数: {len(GLOBAL_MEMORY['PAPER_DB'])}")
        
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
            "description": "包含 id, title, abstract 的字典列表",
            "nullable": True  # 允许直接从全局内存读取数据
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

    def forward(self, raw_papers: list = None) -> list:
        if raw_papers is None:
            # 直接从共享内存“白嫖”数据
            raw_papers = GLOBAL_MEMORY.get("retrieved_papers", [])
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


class PaperDownloaderTool(Tool):
    name = "tool_paper_downloader"
    description = "批量下载PDF文献到本地。传入大纲数据，自动提取大纲中的文献ID，匹配URL并执行下载。"
    inputs = {
        "outline_data": {
            "type": "any",
            "description": "包含文献ID的任意大纲数据结构（字典或列表）"
        },
        "save_dir": {
            "type": "string",
            "description": "保存PDF的本地目录路径"
        },
        "paper_db": {
            "type": "any",
            "description": "可选。包含文献元数据的字典或列表。如果不传，将自动读取共享变量 GLOBAL_MEMORY 中的 PAPER_DB",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, outline_data, save_dir: str, paper_db=None) -> str:
        # 兼容原有逻辑 尝试读取全局共享变量
        if paper_db is None:
            try:
                # 假设 GLOBAL_MEMORY 存在于全局命名空间中
                paper_db = GLOBAL_MEMORY.get("PAPER_DB", {})
            except NameError:
                print("[Downloader] 警告 未传入 paper_db 且未找到全局变量 GLOBAL_MEMORY")
                paper_db = {}

        # 统一化 paper_db 格式 支持列表或字典
        db_dict = {}
        if isinstance(paper_db, list):
            for p in paper_db:
                if isinstance(p, dict) and "id" in p:
                    db_dict[p["id"]] = p
        elif isinstance(paper_db, dict):
            db_dict = paper_db
            
        os.makedirs(os.path.join(save_dir, "pdfs"), exist_ok=True)
        
        # 递归提取大纲中所有的文献 ID
        ids_to_download = set()
        def extract_ids(node):
            if isinstance(node, dict):
                if "supporting_papers" in node and isinstance(node["supporting_papers"], list):
                    for pid in node["supporting_papers"]:
                        if isinstance(pid, str) and pid.strip():
                            ids_to_download.add(pid.strip())
                for v in node.values():
                    extract_ids(v)
            elif isinstance(node, list):
                for item in node:
                    extract_ids(item)

        extract_ids(outline_data)
        print(f"[Downloader] 扫描大纲发现 {len(ids_to_download)} 篇需引用的文献，开始静默下载...")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        success_count = 0
        for pid in ids_to_download:
            paper_info = db_dict.get(pid)
            if not paper_info: 
                continue
                
            # ==========================================
            # 🚀 补丁：保护本地注入的文献，防止被误杀为 online_only
            # ==========================================
            if paper_info.get("status") == "local_success" and paper_info.get("local_path"):
                import os # 确保 os 模块可用
                if os.path.exists(paper_info["local_path"]):
                    success_count += 1
                    print(f"  [+] 命中本地免下载文献: {pid}")
                    continue  # 直接跳过，不走后面的 http 校验和下载逻辑
            # ==========================================
            
            # 兼容不同字段名
            url = paper_info.get("pdf_url") or paper_info.get("url")
            
            # 这里的 http 校验原来会误杀本地文件，现在因为上面的 continue，本地文件不会走到这里了
            if not url or not str(url).startswith("http") or url.lower().endswith(('.jpg', '.png', '.gif')):
                # 更新全局字典状态
                if isinstance(paper_db, dict) and pid in paper_db:
                    paper_db[pid]["status"] = "online_only"
                continue

            safe_filename = "paper_" + hashlib.md5(url.encode()).hexdigest()[:8] + ".pdf"
            local_path = os.path.join(save_dir, "pdfs", safe_filename)

            # 缓存命中逻辑
            if os.path.exists(local_path):
                if isinstance(paper_db, dict) and pid in paper_db:
                    paper_db[pid]["local_path"] = local_path
                    paper_db[pid]["status"] = "local_success"
                success_count += 1
                continue

            success = False
            for attempt in range(2):
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                        if 'text/html' in response.headers.get('Content-Type', '').lower():
                            break 
                            
                        with open(local_path, 'wb') as f:
                            f.write(response.read())
                        success = True
                        break 
                except Exception:
                    time.sleep(1)
            
            # 更新状态回共享变量
            if isinstance(paper_db, dict) and pid in paper_db:
                if success:
                    print(f"  [+] 成功下载并映射 {pid}")
                    paper_db[pid]["local_path"] = local_path
                    paper_db[pid]["status"] = "local_success"
                    success_count += 1
                else:
                    print(f"  [-] 仅保留在线引用 {pid}")
                    paper_db[pid]["status"] = "online_only"

        return f"文献库静默更新完毕。共处理 {len(ids_to_download)} 个ID，成功落盘或命中本地 {success_count} 篇PDF。"
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


class LoadFileTool(Tool):
    name = "load_file"
    description = "从本地读取文件内容。支持读取为纯文本字符串，或自动解析并返回为 JSON 字典/列表。"
    inputs = {
        "file_path": {
            "type": "string",
            "description": "要读取的本地文件绝对或相对路径（例如：'./data/outline.json'）。"
        },
        "as_json": {
            "type": "boolean",
            "description": "是否将文件内容解析为 JSON (字典/列表)。如果为 true，返回字典/列表；如果为 false，返回纯文本字符串。默认为 false。",
            "nullable": True
        }
    }
    output_type = "any" # 因为可能返回 str，也可能返回 dict/list

    def forward(self, file_path: str, as_json: bool = False):
        if not os.path.exists(file_path):
            return f"❌ Error: 文件不存在 - {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 如果大模型指定需要 JSON 解析
            if as_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    return f"❌ Error: JSON 解析失败，文件可能不是标准 JSON 格式 - {str(e)}"
            
            # 否则原样返回纯文本字符串
            return content
            
        except Exception as e:
            return f"❌ Error: 读取文件时发生未知异常 - {str(e)}"

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


class FlattenOutlineTool(Tool):
    name = "tool_flatten_outline"
    description = "将只包含文献ID的树状大纲，结合全局文献库，展平为带有物理路径的写作任务列表。"
    inputs = {
        "outline_data": {
            "type": "any",
            "description": "解析后的大纲JSON对象"
        }
    }
    output_type = "any"

    def forward(self, outline_data) -> dict:
        tasks = []
        main_title = "深度研究学术报告"
        paper_db = GLOBAL_MEMORY.get("PAPER_DB", {})
        
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
            
            # 如果是叶子节点，开始“查表”组装
            if "supporting_papers" in node and isinstance(node["supporting_papers"], list):
                local_papers = []
                online_urls = []
                
                for pid in node["supporting_papers"]:
                    if not isinstance(pid, str): continue
                    
                    # 🚀 核心：通过 ID 从全局数据库反查状态和路径
                    meta = paper_db.get(pid, {})
                    if meta.get("status") == "local_success" and meta.get("local_path"):
                        local_papers.append({"id": pid, "local_path": meta["local_path"]})
                    else:
                        # 如果没有成功下载，退化为在线 URL（或者保留原始 ID 兜底）
                        fallback_url = meta.get("pdf_url") or meta.get("url") or pid
                        online_urls.append(fallback_url)
                        
                tasks.append({
                    "type": "section",
                    "depth": depth,
                    "title": title,
                    "core_argument": node.get("core_argument", ""),
                    "local_papers": local_papers,
                    "online_urls": online_urls
                })
            else:
                # 纯标题节点
                if title or node.get("core_argument"):
                    tasks.append({
                        "type": "heading", 
                        "depth": depth, 
                        "title": title, 
                        "core_argument": node.get("core_argument", "")
                    })
                
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
                print(f"  [{idx}/{len(tasks)}] ✍️ 正在基于SOP框架撰写深度分析初稿...")
                
                # 🚀 这里是核心修改区：通过 SOP 强迫模型输出时间线、横向对比和深度总结
                prompt = f"""你是一名顶会论文的资深学术主笔。请基于以下RAG语料，论证给定的核心论点。

【待撰写章节】: {title}
【本节核心论点】: {core_arg}

【RAG 精准提取语料】:
{rag_context}

【学术写作 SOP (标准作业程序)】
请严格按照以下“四段式”逻辑进行结构化撰写。不要再加小标题了，不过可以在保证写作流畅度的前提下分段落撰写，但必须包含这四个维度的深刻论述：

1. 核心论述 (Core Argument)：开门见山地阐述本节的核心观点与研究背景。
2. 时间演进与趋势分析 (Chronological Trend)：提取语料中的【年份】信息，梳理该技术/问题是如何随时间演变的（例如：“早期研究主要集中于...但近年来逐渐转向...”）。
3. 核心方法横向对比 (Comparative Analysis)：精准提取语料中不同【学者/机构】提出的【具体方法】或【模型】，进行优劣势或【实验数据/指标】的横向对比（例如：“A方法提升了X指标，而B框架则在Y场景下表现更佳”）。
4. 深度总结与洞察 (Synthesis & Insight)：结合上述对比，用一到两句话提炼该方向当前的本质局限或未来的破局点。

【写作强制红线】:
1. 严禁重复输出章节标题：代码引擎会自动加标题，你必须**直接从正文的第一句话开始写**，绝不允许在开头输出 "{title}"。
2. 引用规范：必须在引用数据的句子末尾严格使用提供的 [REF:xxx] 格式。
3. 数据保真：绝不凭空捏造年份、数值或方法。如果语料中缺乏某个维度的数据（如缺乏时间跨度），宁可深挖单一原理，也绝不产生幻觉。
4. 纯净输出：严禁抄袭语料原文自带的数字标号（如 [63]），严禁在文末自行生成参考文献表。
5. 你收到的检索语料来自多篇不同的文章，每篇文章最多只提供了 2 个片段。请综合使用尽可能多的不同文章中的信息来回答问题，不要只依赖同一篇文章的 1–2 个片段。引用越多不同文章，回答的覆盖面就越完整。
"""
                
                try:
                    res = self.writer_agent.run(prompt)
                    draft_content = getattr(res, 'content', str(res))
                except Exception as e:
                    draft_content = f"撰写失败: {str(e)}"

                # 3. 呼叫大模型进行逻辑润色与引用校验
                print(f"  [{idx+1}/{len(tasks)}] 🔄 正在执行逻辑衔接与引用校验...")
                polish_prompt = f"""你是一名严谨的学术审校专家。请对下面的初稿进行润色与事实校验。
【上下文信息】
上一段落内容：{previous_content if previous_content else "（无）"}
本段核心论点：{core_arg}

【RAG 原始基准语料】（用于事实核对）：
{rag_context}

【待润色初稿】
{draft_content}

【必须严格执行的润色指令】
1. 剥离多余格式：直接输出正文文本！如果初稿开头带有类似 `# 1.1 xxx` 或 `## xxx` 的标题，请彻底删除它们。正文中所有的小节分隔一律改用加粗（如 **时间演进**：），绝不允许出现 `#` 号。
2. 逻辑重构：平滑衔接上一段。绝不使用生硬的“机器味”过渡语。
3. 引用纠错：确保文中只出现 [REF:xxx] 格式的引用。彻底删除大模型照抄的无关标号（如 [63]）或自行生成的参考列表。
4. 保留硬核数据：绝对不能在润色时把初稿中提到的作者名、方法名、实验数值和百分比等关键信息删掉。必须保留学术浓度！
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

class GenerateBibliographyTool(Tool):
    name = "tool_generate_bibliography"
    description = "扫描正文中的 [REF:xxx] 标签，替换为 [1][2] 格式，并在文末生成标准的学术参考文献列表。"
    
    # 兼容 tasks_data 废弃参数
    inputs = {
        "report_content": {"type": "string", "description": "全文Markdown"},
        "tasks_data": {"type": "any", "description": "废弃参数", "nullable": True}
    }
    output_type = "string"

    def forward(self, report_content: str, tasks_data=None) -> str:
        import re
        
        # 1. 构建元数据池 (核心修复区)
        meta_lookup = {}
        
        # 融合你之前用过的 retrieved_papers
        for p in GLOBAL_MEMORY.get("retrieved_papers", []):
            pid = p.get("id", "")
            if pid: meta_lookup[pid] = p
            pdf_url = p.get("pdf_url", "")
            if pdf_url: meta_lookup[pdf_url] = p
            
        # 🚀 核心：融合你 RAG 工具里真正存数据的 PAPER_DB
        if "PAPER_DB" in GLOBAL_MEMORY:
            for k, v in GLOBAL_MEMORY["PAPER_DB"].items():
                meta_lookup[k] = v
                if "id" in v:
                    meta_lookup[v["id"]] = v
                    
        # 2. 匹配文中的所有方括号引用
        citations = re.findall(r'\[([^\]]+)\]', report_content)
        unique_refs = []
        ref_mapping = {} 

        for cite in citations:
            sub_cites = [c.strip() for c in cite.split(",")]
            for sc in sub_cites:
                clean_sc = sc.replace("REF:", "").strip() 
                if clean_sc.startswith("http") or clean_sc in meta_lookup or clean_sc.startswith("paper_") or "openalex" in clean_sc:
                    if clean_sc not in unique_refs:
                        unique_refs.append(clean_sc)
                    ref_mapping[sc] = unique_refs.index(clean_sc) + 1
                elif sc.startswith("REF:"): 
                    if clean_sc not in unique_refs:
                        unique_refs.append(clean_sc)
                    ref_mapping[sc] = unique_refs.index(clean_sc) + 1

        # 3. 替换正文标记为数字序号
        def replace_func(match):
            cite_str = match.group(1)
            sub_cites = [c.strip() for c in cite_str.split(",")]
            nums = []
            for sc in sub_cites:
                if sc in ref_mapping:
                    nums.append(str(ref_mapping[sc]))
            if nums: 
                return f"[{','.join(nums)}]"
            return match.group(0)

        formatted_content = re.sub(r'\[([^\]]+)\]', replace_func, report_content)

        # 4. 生成 IEEE 标准排版的参考文献表
        if unique_refs:
            formatted_content += "\n\n---\n\n## 参考文献\n\n"
            for i, ref_id in enumerate(unique_refs):
                num_label = i + 1
                meta = meta_lookup.get(ref_id, {})
                
                # 提取基础字段
                author = meta.get("author", "").strip()
                title = meta.get("title", str(ref_id)).strip()
                year = meta.get("year", "").strip()
                venue = meta.get("venue", "").strip()
                
                # 如果你在之前的工具里保存了 standard_citation (标准格式)，直接用
                if "standard_citation" in meta and meta["standard_citation"]:
                    formatted_content += f"[{num_label}] {meta['standard_citation']}\n"
                    continue
                
                # 否则开始执行动态 IEEE 格式组装
                cite_parts = []
                
                if author and author not in ["未知作者", "Unknown Author", "Unknown"]:
                    cite_parts.append(f"{author}.")
                
                if not title.startswith("http"):
                    title = title.rstrip('.')
                    cite_parts.append(f'"{title}."')
                else:
                    title = "" 
                    
                if venue:
                    cite_parts.append(f"*{venue}*,")
                    
                if year and year not in ["N/A", "Unknown", ""]:
                    cite_parts.append(f"{year}.")
                    
                # 链接处理 (彻底去掉多余的"取自:")
                if str(ref_id).startswith("http"):
                    cite_parts.append(str(ref_id))
                elif meta.get("pdf_url") and meta.get("pdf_url").startswith("http"):
                    cite_parts.append(meta.get("pdf_url"))

                # 防御性合并
                if not cite_parts:
                    cite_parts = [str(ref_id)]

                formatted_content += f"[{num_label}] {' '.join(cite_parts)}\n"

        return formatted_content
import os
import subprocess
# 确保文件顶部有从 smolagents (或你使用的框架) 导入 Tool 的语句

class PDFVisionParserTool(Tool):
    name = "tool_vision_parse"
    description = "调用独立的视觉解析引擎（运行在独立的Python环境中），将PDF文献转化为图文硬绑定的CSV。该任务为计算密集型，耗时较长，请耐心等待其完成。"
    inputs = {
        "input_path": {
            "type": "string",
            "description": "需要解析的PDF文件或PDF所在文件夹的绝对路径"
        },
        "output_path": {
            "type": "string",
            "description": "解析结果(CSV和图片)保存的目标文件夹绝对路径"
        }
    }
    output_type = "string"

    def forward(self, input_path: str, output_path: str) -> str:
        # ==========================================
        # 🚀 核心一：跨环境隔离的秘密武器
        # ==========================================
        # 直接指定“碎纸机”专属的 Python 解释器绝对路径。
        # 这里我根据你之前的报错日志，提取了你 Mac 上的 pdf2md 环境路径。
        # 【注意】：未来给客户部署时，这里要改成客户服务器上那个独立环境的 python 路径
        python_exe = "/Users/wangchao/miniconda3/envs/pdf2md/bin/python" 

        # ==========================================
        # 🚀 核心二：定位启动脚本
        # ==========================================
        # 假设你的 open_deep_research 和 1pdf2md 在同一个父级目录下
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        parser_script = os.path.join(workspace_root, "1pdf2md", "start.py")

        if not os.path.exists(python_exe):
            return f"❌ 严重错误：找不到视觉引擎的隔离Python环境，请检查路径: {python_exe}"
        
        if not os.path.exists(parser_script):
            return f"❌ 严重错误：找不到碎纸机脚本，请检查路径: {parser_script}"

        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n[Command Center] 🚀 跨环境唤醒重型碎纸机中...")
        print(f"  [-] 使用环境: {python_exe}")
        print(f"  [-] 处理目标: {input_path}")
        print(f"  [!] 正在进行视觉版面分析，请耐心等待...\n")
        
        try:
            # ==========================================
            # 🚀 核心三：阻塞等待与执行
            # ==========================================
            # subprocess.run 默认就是阻塞的，主 Agent 会在这里挂起，直到 OCR 彻底跑完
            vision_engine_dir = os.path.dirname(parser_script)
            result = subprocess.run(
                [python_exe, parser_script, "-f", input_path, "-o", output_path],
                capture_output=True,
                text=True,
                check=True,
                cwd=vision_engine_dir  # 🚀 灵魂一步：强制子进程的工作目录切换到 1pdf2md 下！
            )
            return f"✅ 视觉解析全面完成！结构化数据和裁剪好的图片已成功存入 {output_path}。"
            
        except subprocess.CalledProcessError as e:
            # 如果 OCR 过程中报错了（比如之前遇到的缺字体、缺包），这里能精准捕获并反馈给大模型
            error_msg = e.stderr if e.stderr else e.stdout
            print(f"⚠️ 视觉引擎崩溃:\n{error_msg}")
            return f"❌ 视觉引擎执行失败，部分或全部解析未完成。错误日志摘要: {error_msg[-500:]}"
        except Exception as e:
            return f"❌ 发生未知系统调用错误: {str(e)}"
class GenerateAbstractTool(Tool):
    name = "tool_generate_abstract"
    description = "根据已撰写的报告全文和提示词生成【摘要】，并自动将其插入到全局共享内存的报告最开头。"
    inputs = {
        "report_content": {"type": "string", "description": "已撰写的报告全文文本"},
        "prompt": {"type": "string", "description": "摘要的撰写提示词（如：字数限制、必须包含的核心突破等）"}
    }
    output_type = "string"

    def forward(self, report_content: str, prompt: str) -> str:
        # 获取大模型实例
        model = common_utils.ModelProvider.get_model()
        
        # 组装 Prompt
        full_prompt = f"""你是一名资深的学术总编。请根据以下报告全文和特定指令撰写一份高质量的摘要。

【报告全文】：
{report_content}

【撰写指令】：
{prompt}

请直接输出摘要的正文内容，不需要再重复输出“摘要”作为大标题。"""

        messages = [{"role": "user", "content": full_prompt}]
        
        try:
            response = model(messages)
            abstract_content = getattr(response, 'content', str(response)).strip()
        except Exception as e:
            abstract_content = f"摘要生成失败: {e}"

        formatted_abstract = f"## 摘要\n\n{abstract_content}"

        # 1. 存入 GLOBAL_MEMORY["report_list"] (插入到最开头)
        if "report_list" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["report_list"] = []
            
        GLOBAL_MEMORY["report_list"].insert(0, {
            "idx": -1, 
            "title": "摘要", 
            "content": formatted_abstract, 
            "core_arg": "全文摘要", 
            "rag_context": "无", 
            "local_papers": [], 
            "online_urls": []
        })

        # 2. 存入 GLOBAL_MEMORY["final_report"] (拼接在最开头)
        if "final_report" in GLOBAL_MEMORY:
            # 去除原有的 "# 深度研究学术报告" 之类的主标题，可以把摘要插在主标题下，或者直接拼在前头
            # 为了简单健壮，这里直接作为段落插在全文最前面
            GLOBAL_MEMORY["final_report"] = formatted_abstract + "\n\n" + GLOBAL_MEMORY["final_report"]
        else:
            GLOBAL_MEMORY["final_report"] = formatted_abstract

        print("  ✅ 摘要已成功生成并挂载到报告最开头！")
        
        # ❌ 原代码：只返回了摘要
        # return formatted_abstract 
        
        # ✅ 修改为：返回拼接完整的全部报告
        return GLOBAL_MEMORY["final_report"]


class GenerateConclusionTool(Tool):
    name = "tool_generate_conclusion"
    description = "根据已撰写的报告全文和提示词生成【结论】，并自动将其追加到全局共享内存的报告最末尾。"
    inputs = {
        "report_content": {"type": "string", "description": "已撰写的报告全文文本"},
        "prompt": {"type": "string", "description": "结论的撰写提示词（如：要求包含未来展望、局限性等）"}
    }
    output_type = "string"

    def forward(self, report_content: str, prompt: str) -> str:
        # 获取大模型实例
        model = common_utils.ModelProvider.get_model()
        
        # 组装 Prompt
        full_prompt = f"""你是一名资深的学术总编。请根据以下报告全文和特定指令撰写一份深刻的结论。

【报告全文】：
{report_content}

【撰写指令】：
{prompt}

请直接输出结论的正文内容，不需要再重复输出“结论”作为大标题。"""

        messages = [{"role": "user", "content": full_prompt}]
        
        try:
            response = model(messages)
            conclusion_content = getattr(response, 'content', str(response)).strip()
        except Exception as e:
            conclusion_content = f"结论生成失败: {e}"

        formatted_conclusion = f"## 结论\n\n{conclusion_content}"

        # 1. 存入 GLOBAL_MEMORY["report_list"] (追加到最末尾)
        if "report_list" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["report_list"] = []
            
        GLOBAL_MEMORY["report_list"].append({
            "idx": len(GLOBAL_MEMORY["report_list"]), 
            "title": "结论", 
            "content": formatted_conclusion, 
            "core_arg": "全文结论与展望", 
            "rag_context": "无", 
            "local_papers": [], 
            "online_urls": []
        })

        # 2. 存入 GLOBAL_MEMORY["final_report"] (追加在最末尾)
        if "final_report" in GLOBAL_MEMORY:
            GLOBAL_MEMORY["final_report"] = GLOBAL_MEMORY["final_report"].rstrip() + "\n\n" + formatted_conclusion
        else:
            GLOBAL_MEMORY["final_report"] = formatted_conclusion

        print("  ✅ 结论已成功生成并挂载到报告末尾！")
        
        # ❌ 原代码：只返回了结论
        # return formatted_conclusion 
        
        # ✅ 修改为：返回拼接完整的全部报告
        return GLOBAL_MEMORY["final_report"]
class MdToWordTool(Tool):
    name = "tool_md_to_word"
    description = "将 Markdown 格式的长文文本转换为标准的 Word (.docx) 文档并保存。适合在学术报告定稿后调用，生成可供直接提交的格式。"
    inputs = {
        "md_content": {
            "type": "string",
            "description": "要转换的完整 Markdown 文本内容。"
        },
        "out_dir": {
            "type": "string",
            "description": "输出目录路径（通常是 run_dir）。"
        },
        "file_name": {
            "type": "string",
            "description": "不含后缀的输出文件名（例如 'final_academic_report'）。"
        }
    }
    output_type = "string"

    def forward(self, md_content: str, out_dir: str, file_name: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{file_name}.docx")
        
        try:
            # 核心：将 markdown 字符串直接转换为 docx 文件
            # extra_args 可以确保换行和基础样式被正确解析
            pypandoc.convert_text(
                md_content, 
                'docx', 
                format='md', 
                outputfile=output_path,
                extra_args=['--reference-doc=reference.docx'] if os.path.exists('reference.docx') else []
            )
            return f"✅ 格式转换成功！Word 文档已生成并保存至: {output_path}"
        except Exception as e:
            return f"❌ Word 转换发生异常: {str(e)}\n请确保已通过 'brew install pandoc' 安装了底层依赖。"

class LocalPaperInjectorTool(Tool):
    name = "tool_local_paper_injector"
    description = "扫描指定的本地文件夹，读取其中的全部 PDF 文件。自动提取内容并将这些文献注入到系统的全局内存中，供后续的提纯和生成大纲使用。必须传入绝对或相对的文件夹路径。"
    inputs = {
        "folder_path": {
            "type": "string",
            "description": "存放本地 PDF 文件的文件夹路径（例如：'./local_papers'）"
        }
    }
    output_type = "string"

    def forward(self, folder_path: str) -> str:
        import os
        import fitz  # PyMuPDF
        import re
        
        # 确保全局内存已经初始化
        if "PAPER_DB" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["PAPER_DB"] = {}
        if "retrieved_papers" not in GLOBAL_MEMORY:
            GLOBAL_MEMORY["retrieved_papers"] = []
            
        abs_folder_path = os.path.abspath(folder_path)
            
        if not os.path.exists(abs_folder_path):
            return f"❌ 错误：找不到文件夹绝对路径 {abs_folder_path}"

        pdf_files = [f for f in os.listdir(abs_folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return f"⚠️ 警告：在 {abs_folder_path} 中没有找到任何 PDF 文件。"

        print(f"[LocalInjector] 发现 {len(pdf_files)} 篇本地 PDF，开始提取元数据与精准摘要...")
        
        success_count = 0
        for idx, filename in enumerate(pdf_files):
            abs_file_path = os.path.join(abs_folder_path, filename)
            paper_id = f"local_paper_{idx}"
            
            try:
                title = filename.replace(".pdf", "")
                author = "未知作者"
                year = "未知年份"
                final_abstract = ""
                
                with fitz.open(abs_file_path) as doc:
                    meta = doc.metadata
                    
                    # 1. 提取元数据 (标题、作者、年份)
                    if meta.get("title") and meta.get("title").strip():
                        title = meta.get("title").strip()
                    if meta.get("author") and meta.get("author").strip():
                        author = meta.get("author").strip()
                        
                    creation_date = meta.get("creationDate", "")
                    if creation_date.startswith("D:") and len(creation_date) >= 6:
                        year = creation_date[2:6]
                    else:
                        year_match = re.search(r'(19|20)\d{2}', filename)
                        if year_match:
                            year = year_match.group(0)

                    # ==========================================
                    # 🚀 核心更新：正则精准提取真实摘要
                    # ==========================================
                    # 先读取前两页的全部生肉文本
                    raw_text = ""
                    for page in doc[:2]:
                        raw_text += page.get_text("text") + "\n"
                        
                    # 使用正则匹配：从 "Abstract" 或 "摘要" 开始，到 "Introduction", "引言", "Keywords" 或 "1. " 结束
                    abstract_pattern = r'(?i)(?:abstract|摘\s*要)\s*[:\n]?\s*(.*?)(?:\n\s*(?:introduction|引\s*言|1\.\s|keywords|关键\s*词))'
                    match = re.search(abstract_pattern, raw_text, re.DOTALL)
                    
                    if match and len(match.group(1).strip()) > 50:
                        # 命中正则：获得了非常干净的纯摘要！
                        final_abstract = match.group(1).strip()
                        # 清理掉多余的换行符，让段落连贯
                        final_abstract = re.sub(r'\n+', ' ', final_abstract)
                    else:
                        # 兜底降级方案：如果排版太奇葩没匹配到 Abstract，就砍掉前 1500 个字符充当摘要
                        final_abstract = raw_text[:1500].strip()
                        final_abstract = re.sub(r'\n+', ' ', final_abstract)
                
            except Exception as e:
                print(f"  [-] 解析 {filename} 失败: {e}")
                continue
                
            # 组装动态获取到的论文元数据
            paper_data = {
                "id": paper_id,
                "title": title,
                "abstract": final_abstract if final_abstract else "【无摘要，需依赖RAG正文提取】",
                "local_path": abs_file_path,          
                "status": "local_success",            
                "year": year,                     
                "author": author,                 
                "pdf_url": abs_file_path,             
                "source_query": "local_injection"     
            }
            
            GLOBAL_MEMORY["PAPER_DB"][paper_id] = paper_data
            GLOBAL_MEMORY["retrieved_papers"].append(paper_data)
            success_count += 1
            
        result_msg = f"✅ 成功扫描 {abs_folder_path}，已精准提取 {success_count} 篇本地文献的摘要与元数据！"
        print(f"[LocalInjector] {result_msg}")
        return result_msg