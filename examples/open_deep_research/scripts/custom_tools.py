import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
import ast
import copy
import time
import ssl
import hashlib
import csv
import subprocess
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
from utils.state_manager import ResearchStateManager # 引入刚才写好的管理器
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

state = ResearchStateManager()

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
        # 🚀 1. 引入状态管理器
        
        
        # 提取数据库中已有的文献缓存，方便 O(1) 查找
        all_db_papers = state.get_all_papers()
        # 兼容 paper_id 和 url 作为 key
        paper_db_cache = {p.get("id"): p for p in all_db_papers if p.get("id")}
        
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      
            chunk_overlap=250,    
            separators=["\n\n", "。", "\n", ".", "；"] 
        )
        
        # 2. Query 净化：剥离大模型强加的写作用指令词，提取纯净的学术语义用于检索
        clean_query = query
        stop_words = ["具体方法", "实验数据", "提升指标", "性能对比", "百分比", "具体数值", "详细阐述"]
        for word in stop_words:
            clean_query = clean_query.replace(word, "")
        clean_query = re.sub(r'[^\w\u4e00-\u9fa5a-zA-Z0-9]+', ' ', clean_query).strip()

        print(f"[RAG] 启动混合检索 | 净化Query: {clean_query[:25]}... | PDF: {len(local_papers)}篇 | Web: {len(online_urls)}个")

        # 3. 解析本地 PDF 并汇入总池 (🚀 注入 1pdf2md 视觉解析逻辑)
        for paper in local_papers:
            pid, path = paper.get('id'), paper.get('local_path')
            if path and os.path.exists(path):
                # 🚀 数据库交互：如果是不认识的新 PDF，提取元数据并持久化
                if pid not in paper_db_cache:
                    new_meta = {
                        "id": pid,
                        "type": "paper",
                        "title": os.path.basename(path),
                        "author": "Unknown",
                        "year": "N/A"
                    }
                    try:
                        with fitz.open(path) as pdf:
                            meta = pdf.metadata
                            new_meta.update({
                                "title": meta.get("title") or os.path.basename(path),
                                "author": meta.get("author") or "Unknown Author",
                                "year": meta.get("creationDate", "    ")[2:6] if meta.get("creationDate") else "N/A"
                            })
                    except Exception:
                        pass
                        
                    state.upsert_paper(new_meta)
                    paper_db_cache[pid] = new_meta

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

        # 4. 解析在线网页并汇入总池 (包含 OpenAlex API 拦截和 SSL 绕过)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        for url in online_urls:
            # 🚀 将 URL 视为一种特殊的 paper_id 进行持久化
            if url not in paper_db_cache:
                new_url_meta = {"id": url, "type": "web", "title": url, "url": url}
                state.upsert_paper(new_url_meta)
                paper_db_cache[url] = new_url_meta

            if url not in WEB_CACHE:
                chunks = []
                is_processed = False

                # 【新增修复】：不管是什么引擎的 ID，只要数据库里有摘要，就直接提取！
                matched_paper = paper_db_cache.get(url)
                if matched_paper and matched_paper.get("abstract") and matched_paper.get("abstract") not in ["No abstract available.", ""]:
                    title = matched_paper.get("title", url)
                    text_content = f"【论文标题】: {title}\n【完整摘要】: {matched_paper['abstract']}"
                    for c in splitter.split_text(text_content):
                        chunks.append({"source_id": url, "page": "文献摘要(内存)", "text": c})
                    is_processed = True
                if not is_processed:
                    # 尝试正则匹配 OpenAlex 的 Work ID
                    oa_match = re.search(r'(W\d{8,})', url)
                    
                    # 🚀 修复点：增加判空保护！只有成功匹配到 W 开头的 ID，才进入 OpenAlex 的专属逻辑
                    if oa_match:
                        work_id = oa_match.group(1)
                        matched_paper = paper_db_cache.get(work_id)
                        
                        if matched_paper and matched_paper.get("abstract") and matched_paper.get("abstract") != "No abstract available.":
                            title = matched_paper.get("title", url)
                            # 同步更新到数据库
                            paper_db_cache[url].update({
                                "title": title,
                                "author": matched_paper.get("author", "未知学者"),
                                "year": str(matched_paper.get("year", "N/A"))
                            })
                            state.upsert_paper(paper_db_cache[url])
                            
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
                                    
                                    # 同步更新到数据库
                                    paper_db_cache[url].update({"title": title, "author": author, "year": year})
                                    state.upsert_paper(paper_db_cache[url])
                                    
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
                                    # 🚀 同步更新到数据库
                                    paper_db_cache[url]["title"] = line.replace("Title: ", "").strip()
                                    state.upsert_paper(paper_db_cache[url])
                                    break
                            for c in splitter.split_text(content):
                                chunks.append({"source_id": url, "page": "网页正文", "text": c})
                    except Exception as e:
                        pass
                        
                WEB_CACHE[url] = chunks
            
            all_chunks.extend(WEB_CACHE.get(url, []))

        if not all_chunks: 
            return "未提取到可用文本，请直接基于常识撰写。"

        # 5. 混合池双路检索与重排
        tokenized_corpus = [list(c["text"]) for c in all_chunks]
        tokenized_query = list(clean_query.replace(" ", ""))
        bm25_scores = BM25Okapi(tokenized_corpus).get_scores(tokenized_query)
        
        c_emb = EMBEDDER.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
        q_emb = EMBEDDER.encode(clean_query, normalize_embeddings=True)
        vec_scores = np.dot(c_emb, q_emb)
        
        bn = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        vn = (vec_scores - np.min(vec_scores)) / (np.max(vec_scores) - np.min(vec_scores) + 1e-9)
        hybrid = 0.4 * bn + 0.6 * vn
        
        candidates = []
        pre_source_counts = {}
        
        # 遍历所有按混合分数排好序的片段
        for idx in np.argsort(hybrid)[::-1]:
            c = all_chunks[idx]
            sid = c["source_id"]
            
            # 限制单篇文献进入重排池的片段数（比如最多只允许 4 个进入重排）
            if pre_source_counts.get(sid, 0) < 4:
                candidates.append(c)
                pre_source_counts[sid] = pre_source_counts.get(sid, 0) + 1
            
            # 收集足够多的多样化候选后再去重排（扩大候选池到 50 个）
            if len(candidates) >= 50:
                break
        
        cross_inp = [[clean_query, c["text"]] for c in candidates]
        rerank_scores = RERANKER.predict(cross_inp)
        
        # 6. 核心打散机制：强制同源去重
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
        
        # 7. 组装最终喂给大模型的语料 (🚀 直接从缓存字典读取元数据)
        res = f"【检索查询】: {query}\n\n" 
        for i, c in enumerate(final_chunks):
            ref_id = c['source_id']
            # 从重构的局部缓存直接拿取元数据
            meta = paper_db_cache.get(ref_id, {})
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

# =====================================================================
# 工具 1：学术广度搜索工具 (The Foraging Tool)
# =====================================================================

class AcademicSearchTool(Tool):
    name = "tool_academic_search"
    description = "用于在 ArXiv, OpenAlex 或 Semantic Scholar 上检索学术论文。输入英文关键词，返回论文基础信息的 JSON 列表。会自动将检索结果双向同步到全局内存 `retrieved_papers` (列表) 和 `PAPER_DB` (字典字典) 中。"
    inputs = {
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "英文关键词列表"
        },
        "engine": {
            "type": "string", 
            "description": "'arxiv', 'openalex' 或 'semanticscholar'", 
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
        # (保留你原本的环境初始化逻辑)
        engine = (engine or "openalex").lower()
        sort_by = (sort_by or "date").lower()
        max_results = max_results_per_query or 10
        all_papers = {} 
        all_papers_this_batch = []
        
        existing_papers = state.get_all_papers()
        existing_titles = {p.get("title", "").strip().lower() for p in existing_papers if p.get("title")}
        
        # 🚀 寻找当前数据库中 p+数字 系列的最大值
        max_p_id = 0
        for p in existing_papers:
            pid = p.get("id", "")
            if pid.startswith("p") and pid[1:].isdigit():
                max_p_id = max(max_p_id, int(pid[1:]))
        current_p_id = max_p_id
        
        for query in search_queries:
            try:
                if engine == "openalex":
                    # ==========================================
                    # OpenAlex 抗压与防封禁优化方案
                    # ==========================================
                    encoded_query = urllib.parse.quote(query.strip())
                    sort_param = "cited_by_count:desc" if sort_by == "citation" else "relevance_score:desc"
                    # 强制加入 mailto 进入高优礼貌池
                    url = f"https://api.openalex.org/works?search={encoded_query}&sort={sort_param}&per-page={max_results}&mailto=wangchao@example.com"
                    
                    max_retries = 3
                    success = False
                    
                    for attempt in range(max_retries):
                        try:
                            headers = {'User-Agent': 'open_deep_research_agent/1.0 (mailto:wangchao@example.com)'}
                            req = urllib.request.Request(url, headers=headers)
                            with urllib.request.urlopen(req, timeout=15) as response:
                                data = json.loads(response.read().decode('utf-8'))
                                for paper in data.get('results', []):
                                    title = paper.get('title')
                                    # 🚀 数据库级去重拦截 + 批次级去重拦截
                                    if not title or title.strip().lower() in existing_titles or title in all_papers: 
                                        continue

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
                                    current_p_id += 1
                                    new_pid = f"p{current_p_id}"
                                    
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
                                        "id": new_pid,
                                        "original_id": oa_id,
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
                            success = True
                            break # 成功则跳出重试循环
                            
                        except urllib.error.HTTPError as e:
                            print(f"  [OpenAlex API] 第 {attempt+1} 次请求失败: HTTP {e.code} ({query[:20]}...)")
                            if e.code == 429: # 针对并发超限专门处理
                                time.sleep(5)
                            else:
                                time.sleep(2)
                        except Exception as e:
                            print(f"  [OpenAlex API] 第 {attempt+1} 次网络层异常 (可能断流): {e}")
                            time.sleep(3)
                            
                    if not success:
                        print(f"  ⚠️ 检索词 '{query[:30]}...' 连续 {max_retries} 次获取失败，已跳过。")
                        
                    time.sleep(2)
                    
                elif engine in ["semanticscholar", "s2"]:
                    # ==========================================
                    # Semantic Scholar 引擎接入 (带环境变量 S2_API_KEY)
                    # ==========================================
                    
                    # 🚀 修复点1：保留双引号以支持精确匹配，只清洗会引发 S2 报错的复杂布尔符号
                    safe_query = query.replace(" AND ", " ").replace(" OR ", " ").replace("(", "").replace(")", "")
                    
                    # 🚀 修复点2：拦截中文，防止 S2 搜索崩溃返回随机 Review
                    if any('\u4e00' <= char <= '\u9fff' for char in safe_query):
                        print(f"  [Semantic Scholar API] ⚠️ 警告：检测到中文 '{safe_query}'，S2 API 不支持中文检索，可能会返回无关结果！")
                        # 可选：你可以直接 return 或者抛出异常
                        
                    encoded_query = urllib.parse.quote(safe_query.strip())
                    # 注意：S2 的默认排序是 relevance，如果要按时间，应为 year:desc
                    sort_param = "&sort=citationCount:desc" if sort_by == "citation" else "&sort=year:desc" if sort_by == "date" else ""
                    fields = "title,authors,year,abstract,openAccessPdf,venue,url"
                    
                    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={max_results}&fields={fields}{sort_param}"
                    api_key = os.environ.get("S2_API_KEY", "")
                    
                    max_retries = 3
                    success = False
                    
                    for attempt in range(max_retries):
                        try:
                            headers = {'User-Agent': 'open_deep_research_agent/1.0'}
                            if api_key:
                                headers['x-api-key'] = api_key
                                
                            req = urllib.request.Request(url, headers=headers)
                            with urllib.request.urlopen(req, timeout=15) as response:
                                data = json.loads(response.read().decode('utf-8'))
                                for paper in data.get('data', []):
                                    title = paper.get('title')
                                    if not title or title.strip().lower() in existing_titles or title in all_papers: 
                                        continue
                                    
                                    authors_list = [a.get('name', '') for a in paper.get('authors', [])]
                                    author_str = ", ".join(authors_list) if authors_list else "Unknown Author"
                                    author_display = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "") if authors_list else "Unknown Author"
                                    
                                    year = str(paper.get('year', 'Unknown'))
                                    s2_id = paper.get('paperId', '')
                                    current_p_id += 1
                                    new_pid = f"p{current_p_id}"
                                    abstract = paper.get('abstract') or "No abstract available."
                                    venue_name = paper.get('venue', '')
                                    
                                    pdf_url = ""
                                    oa_info = paper.get('openAccessPdf')
                                    if oa_info and isinstance(oa_info, dict):
                                        pdf_url = oa_info.get('url', '')
                                        
                                    std_cite = f'{author_display}. "{title}."'
                                    if venue_name: std_cite += f' *{venue_name}*,'
                                    std_cite += f' {year}. 取自: SemanticScholar {s2_id}'
                                    
                                    all_papers[title] = {
                                        "id": new_pid,
                                        "original_id": s2_id,
                                        "title": title,
                                        "year": year,
                                        "author": author_str,
                                        "abstract": abstract,
                                        "pdf_url": pdf_url,
                                        "source_query": query,
                                        "venue": venue_name,
                                        "standard_citation": std_cite,
                                    }
                            success = True
                            break 
                            
                        except urllib.error.HTTPError as e:
                            print(f"  [Semantic Scholar API] 第 {attempt+1} 次请求失败: HTTP {e.code} ({query[:20]}...)")
                            if e.code == 429: # 限流处理
                                time.sleep(6) # 增加延迟
                            else:
                                time.sleep(2)
                        except Exception as e:
                            print(f"  [Semantic Scholar API] 第 {attempt+1} 次网络层异常: {e}")
                            time.sleep(3)
                            
                    # 🚀 修复点3：请求彻底失败时，必须抛出异常，绝不能静默返回！
                    if not success:
                        error_msg = f"API 检索彻底失败 (连续 {max_retries} 次限流或网络断开)。请检查 S2_API_KEY 是否有效，或降低并发。当前 Query: {query}"
                        print(f"  ⚠️ {error_msg}")
                        raise RuntimeError(error_msg)  # 打断外层流程，防止它塞入“兜底垃圾数据”
                        
                    time.sleep(2)

                else: 
                    # ==========================================
                    # ArXiv 专属抗压优化方案
                    # ==========================================
                    safe_query = query.replace(" AND ", " ").replace(" OR ", " ").replace("(", "").replace(")", "")
                    encoded_query = urllib.parse.quote(f'all:{safe_query.strip()}')
                    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
                    
                    max_retries = 3
                    success = False
                    for attempt in range(max_retries):
                        try:
                            headers = {'User-Agent': 'open_deep_research_agent/1.0 (mailto:wangchao@example.com)'}
                            req = urllib.request.Request(url, headers=headers)
                            
                            with urllib.request.urlopen(req, timeout=15) as response:
                                root = ET.fromstring(response.read().decode('utf-8'))
                                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                                for entry in root.findall('atom:entry', ns):
                                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                                    # 🚀 数据库级去重拦截 + 批次级去重拦截
                                    if not title or title.strip().lower() in existing_titles or title in all_papers: 
                                        continue
                                    
                                    author_elements = entry.findall('atom:author/atom:name', ns)
                                    authors = [a.text.strip() for a in author_elements if a.text]
                                    author_display = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown Author"
                                    
                                    year = entry.find('atom:published', ns).text.split('-')[0]
                                    paper_id = entry.find('atom:id', ns).text.strip()
                                    current_p_id += 1
                                    new_pid = f"p{current_p_id}"
                                    pdf_url = paper_id.replace('/abs/', '/pdf/') + ".pdf" 
                                    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                                    
                                    std_cite = f'{author_display}. "{title}." *arXiv preprint*, {year}. 取自: {paper_id}'
                                    
                                    all_papers[title] = {
                                        "id": new_pid,
                                        "original_id": paper_id,
                                        "title": title,
                                        "year": year,
                                        "author": author_display,
                                        "abstract": abstract,
                                        "pdf_url": pdf_url,
                                        "source_query": query,
                                        "standard_citation": std_cite
                                    }
                            success = True
                            break 
                            
                        except urllib.error.HTTPError as e:
                            print(f"  [ArXiv API] 第 {attempt+1} 次请求失败: HTTP {e.code} ({query[:20]}...)")
                            time.sleep(4) 
                        except Exception as e:
                            print(f"  [ArXiv API] 第 {attempt+1} 次请求发生异常: {e}")
                            time.sleep(2)
                            
                    if not success:
                        print(f"  ⚠️ 检索词 '{query[:30]}...' 连续 {max_retries} 次获取失败，已跳过。")
                        
                    time.sleep(3.5)

            except Exception as e:
                print(f"[AcademicSearchTool] 检索词 '{query}' 发生全局异常: {e}")

        # ====================================================================
        # 🚀 核心架构升级：使用 TinyDB 状态管理器进行持久化入库
        # ====================================================================
        
        for paper_data in all_papers.values():
            if paper_data.get("id"):
                state.upsert_paper(paper_data)
                all_papers_this_batch.append(paper_data)
        
        # 统计数据库最终的总数据量
        total_in_db = len(state.get_all_papers())
        print(f"[AcademicSearchTool] 入库完毕！本次新抓取 {len(all_papers_this_batch)} 篇，本地沙盒累计安全归档 {total_in_db} 篇文献。")
        
        return all_papers_this_batch


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
            paper['core_breakthrough'] = "无"
            paper['limitation'] = "无"
            paper['insight'] = "摘要过短，无明确洞察"
            return paper

        # 🚀 优化 1：强制要求输出严格的 JSON 结构
        prompt = f"""请提炼以下摘要的核心情报。
要求：必须且只能输出一个合法的 JSON 对象，不要使用 Markdown 代码块，包含两个字段：
"core_breakthrough": "用一两句话说明该论文的核心技术突破或新方法",
"limitation": "用一两句话说明该方法存在的局限或试图解决的痛点"
总字数不超过 60 字。
摘要内容：{abstract}"""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = model(messages)
            content = getattr(response, 'content', str(response)).strip()
            
            # 兼容并清洗大模型可能带有的 markdown 代码块符号
            cleaned = re.sub(r'^\`\`\`(?:json)?\n', '', content, flags=re.MULTILINE)
            cleaned = re.sub(r'\`\`\`$', '', cleaned, flags=re.MULTILINE).strip()
            
            # 解析双字段
            parsed = json.loads(cleaned)
            core_breakthrough = parsed.get("core_breakthrough", "未提取出核心突破")
            limitation = parsed.get("limitation", "未提取出局限")
            
            # 组合成原有的 insight 字符串以向下兼容 RAG 引擎的匹配
            insight_str = f"【核心突破】：{core_breakthrough}；【局限/痛点】：{limitation}"
            
        except Exception as e:
            core_breakthrough = "提炼突破失败"
            limitation = "提炼局限失败"
            insight_str = f"提炼失败: {e}"

        return {
            "id": paper.get('id'),
            "title": paper.get('title'),
            "year": paper.get('year'),
            "core_breakthrough": core_breakthrough,  # ✨ 新增的独立突破字段
            "limitation": limitation,                # ✨ 新增的独立局限字段
            "insight": insight_str,                  # 兼容原架构
            "pdf_url": paper.get('pdf_url'),
            "source_query": paper.get('source_query')
        }

    def forward(self, raw_papers: list = None) -> list:
        # 🚀 替换点：使用 TinyDB 获取所有文献
        if not raw_papers:
            
            raw_papers = state.get_all_papers()
            
        # 从全局 common_utils 获取模型实例
        model = common_utils.ModelProvider.get_model()
        compressed_papers = []
        
        print(f"[InsightExtractorTool] 开始并发提炼 {len(raw_papers)} 篇摘要...")
        # 保持多线程加速，smolagents 模型请求(底层 HTTP)一般线程安全
        with ThreadPoolExecutor(max_workers=5) as executor:
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
    description = "将提纯后的文献情报进行动态层次聚类。底层确保每簇10-15篇文献，向上按比例聚合，直到顶层簇数量不大于8个。输出动态层级的聚类树。"
    inputs = {
        "compressed_papers": {
            "type": "array",
            "items": {"type": "object"},
            "description": "包含 insight 字段的文献字典列表"
        }
    }
    output_type = "any"

    def forward(self, compressed_papers: list) -> dict:
        try:
            import umap
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            return {"error": "缺少依赖，请在终端运行: pip install umap-learn scikit-learn"}

        if not compressed_papers:
            return {"clusters": []}

        texts = [f"{p.get('title', '')} {p.get('core_breakthrough', p.get('insight', ''))}" for p in compressed_papers]
        embeddings = EMBEDDER.encode(texts, normalize_embeddings=True)
        
        # 初始化叶子节点（单篇论文）
        current_nodes = [{"type": "paper", "data": p, "embedding": emb} for p, emb in zip(compressed_papers, embeddings)]
        n_papers = len(current_nodes)
        
        print(f"[SemanticClusterTool] 开始动态层次聚类，当前总文献数: {n_papers}")
        
        # ==========================================
        # 1. 强制生成最底层（叶子节点层）：要求平均每簇 12 篇文献
        # ==========================================
        target_leaf_size = 12 
        n_leaf_clusters = max(1, n_papers // target_leaf_size)
        
        if n_leaf_clusters == 1:
            print("  [-] 文献数量较少，直接返回单层簇。")
            return {"clusters": [{"cluster_type": "level_1", "items": [{"paper_id": p["data"]["id"], "title": p["data"]["title"], "insight": p["data"]["insight"]} for p in current_nodes]}]}

        try:
            reducer = umap.UMAP(n_neighbors=min(15, n_papers - 1), n_components=min(5, n_papers-2), random_state=42)
            reduced_emb = reducer.fit_transform(np.array([n["embedding"] for n in current_nodes]))
            kmeans = KMeans(n_clusters=n_leaf_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(reduced_emb)
        except Exception as e:
            print(f"  [!] 第一层降维异常，启用均匀切分 ({e})")
            labels = np.arange(n_papers) % n_leaf_clusters
            
        leaf_clusters = []
        for l in range(n_leaf_clusters):
            children = [current_nodes[i] for i, x in enumerate(labels) if x == l]
            if not children: continue
            leaf_clusters.append({
                "type": "cluster",
                "level_depth": 1, 
                "children": children,
                "embedding": np.mean([n["embedding"] for n in children], axis=0)
            })
            
        print(f"  [-] Level-1 (底层小节) 聚类完成：切分为 {len(leaf_clusters)} 个小节，每节约 {n_papers//n_leaf_clusters} 篇。")

        # ==========================================
        # 2. 动态向上聚合循环：只要当前层的数量 > 8，就继续往上聚
        # ==========================================
        current_level_nodes = leaf_clusters
        depth = 1
        
        while len(current_level_nodes) > 8:
            depth += 1
            n_current = len(current_level_nodes)
            
            # 设定：每 4 个子节点（小节）合并为 1 个父节点（大章）
            n_parent_clusters = max(2, n_current // 4) 
            
            n_neighbors = min(5, n_current - 1)
            if n_neighbors < 2: n_neighbors = 2
            
            try:
                reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=min(3, n_current-2), random_state=42)
                reduced_emb = reducer.fit_transform(np.array([n["embedding"] for n in current_level_nodes]))
                kmeans = KMeans(n_clusters=n_parent_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(reduced_emb)
            except Exception as e:
                print(f"  [!] Level-{depth} 聚类异常，启用退避切分 ({e})")
                labels = np.arange(n_current) % n_parent_clusters
                
            parent_clusters = []
            for l in range(n_parent_clusters):
                children = [current_level_nodes[i] for i, x in enumerate(labels) if x == l]
                if not children: continue
                parent_clusters.append({
                    "type": "cluster",
                    "level_depth": depth,
                    "children": children,
                    "embedding": np.mean([n["embedding"] for n in children], axis=0)
                })
                
            current_level_nodes = parent_clusters
            print(f"  [-] Level-{depth} (上层大章) 聚类完成：聚合为 {len(current_level_nodes)} 个高级簇。")

        print(f"[SemanticClusterTool] 动态组装完毕。这是一棵 {depth} 层深的逻辑树，顶层大章数量: {len(current_level_nodes)}")

        # 3. 递归清理节点树，去掉冗余向量，输出给大模型
        def clean_tree(node):
            if node["type"] == "paper":
                return {
                    "paper_id": node["data"].get("id"),
                    "title": node["data"].get("title"),
                    "core_breakthrough": node["data"].get("core_breakthrough"), # ✨ 暴露核心突破给大纲生成
                    "limitation": node["data"].get("limitation"),               # ✨ 暴露局限给大纲生成
                    "insight": node["data"].get("insight")
                }
            else:
                return {
                    "node_type": "nested_cluster",
                    "items": [clean_tree(c) for c in node["children"]]
                }
                
        return {"clusters": [clean_tree(n) for n in current_level_nodes]}


class SaveFileTool(Tool):
    name = "save_file"
    description = "安全保存文件工具。支持穿透沙盒写文件，可选择覆盖写入或追加写入。"
    
    inputs = {
        "content": {"type": "any", "description": "要保存的内容"},
        "file_name": {"type": "string", "description": "文件名（不含后缀）"},
        "out_type": {"type": "string", "description": "文件后缀类型，如 md 或 json"},
        "append": {
            "type": "boolean", 
            "description": "是否追加写入。True 为追加，False 为覆盖。默认为 False。", 
            "nullable": True
        },
        "out_dir": {"type": "string", "description": "输出目录路径", "nullable": True},
    }
    output_type = "string"
    def __init__(self, base_dir=None):
        super().__init__()
        self.base_dir = base_dir

    def forward(self, content, file_name: str, out_type: str, append: bool = False, out_dir: str = None) -> str:
        if self.base_dir:
            out_dir = os.path.join(self.base_dir, out_dir) if out_dir else self.base_dir
        else:
            out_dir = out_dir or "."
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

    def __init__(self, base_dir=None):
        super().__init__()
        self.base_dir = base_dir

    def forward(self, file_path: str, as_json: bool = False):
        if self.base_dir:
            file_path = os.path.join(self.base_dir, file_path)

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


import os, hashlib, ssl, urllib.request
import numpy as np
from typing import List, Dict, Any, Optional

class FlattenOutlineTool(Tool):
    name = "tool_flatten_outline"
    description = (
        "将逻辑大纲树展开为写作任务列表。"
        "优先使用大纲节点硬挂载的 paper_ids，否则自动进行语义向量匹配。"
        "自动下载在线 PDF 到 save_dir，并记录本地路径。"
    )
    inputs = {
        "outline_data": {
            "type": "any",
            "description": "解析后的大纲 JSON 对象"
        },
        "papers": {
            "type": "any",
            "description": "文献列表，每项为字典，必须包含 'id'。建议包含 'title', 'insight', 'pdf_url' 等字段。"
        },
        "save_dir": {
            "type": "string",
            "description": "保存 PDF 的本地目录，可选。若不传则自动使用 outputs/pdfs",
            "nullable": True
        }
    }
    output_type = "any"

    def forward(
        self,
        outline_data,
        papers: List[Dict[str, Any]],
        save_dir: Optional[str] = None
    ) -> dict:
        tasks = []
        main_title = "深度研究学术报告"

        # 设置保存目录
        if not save_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(base_dir, "outputs", "pdfs")
        os.makedirs(save_dir, exist_ok=True)

        # 解析大纲根节点
        if isinstance(outline_data, dict):
            main_title = outline_data.get("level_1_title") or outline_data.get("chapter_title") or main_title
            nodes = outline_data.get("chapters") or outline_data.get("sub_chapters") or outline_data.get("level_1") or [outline_data]
        elif isinstance(outline_data, list):
            nodes = outline_data
        else:
            nodes = []

        # ---------- 将传入的 papers 转为 paper_db ----------
        paper_db = {}
        for p in papers:
            pid = p.get("id")
            if not pid:
                continue
            # 标准化：确保必要字段存在
            p.setdefault("status", "unknown")
            p.setdefault("local_path", None)
            # 若 pdf_url 是本地路径且存在，直接标记 local_success
            pdf_url = p.get("pdf_url", "")
            if pdf_url and not pdf_url.startswith("http"):
                if os.path.isfile(pdf_url):
                    p["local_path"] = pdf_url
                    p["status"] = "local_success"
            paper_db[pid] = p

        # ---------- 构建向量匹配所需数据 ----------
        candidate_docs = []
        valid_pids = []
        for pid, p in paper_db.items():
            text_rep = f"【标题】{p.get('title', '')} 【内容】{p.get('insight', p.get('abstract', ''))}"
            candidate_docs.append(text_rep)
            valid_pids.append(pid)

        doc_embeddings = None
        if candidate_docs:
            doc_embeddings = EMBEDDER.encode(candidate_docs, normalize_embeddings=True)

        global_paper_usage = {}

        # ---------- 递归遍历大纲树 ----------
        def traverse(node, depth):
            if not isinstance(node, dict):
                return

            title = node.get("chapter_title") or node.get("section_title") or node.get("subsection_title") or ""
            core_arg = node.get("core_argument", "")
            explicit_paper_ids = node.get("paper_ids", [])  # 大纲节点直接指定的文献 ID

            children_keys = ["chapters", "sections", "sub_chapters", "subsections", "level_2", "children"]
            has_children = any(k in node and isinstance(node[k], list) for k in children_keys)

            if core_arg and not has_children:  # 叶子节点 -> 生成写作任务
                local_papers = []
                online_urls = []
                target_pids = []

                # 优先使用硬挂载的 paper_ids
                if explicit_paper_ids:
                    print(f"  [*] 章节 '{title}' -> 命中聚类物理关联，硬挂载文献 IDs: {explicit_paper_ids}")
                    target_pids = [pid for pid in explicit_paper_ids if pid in paper_db]
                elif doc_embeddings is not None:
                    print(f"  [⚠️] 章节 '{title}' 未挂载文献，启动全局语义向量匹配...")
                    query_text = f"{title} {core_arg}"
                    query_emb = EMBEDDER.encode(query_text, normalize_embeddings=True)
                    scores = np.dot(doc_embeddings, query_emb)

                    # 惩罚已使用文献，促进多样性
                    for i, pid in enumerate(valid_pids):
                        usage = global_paper_usage.get(pid, 0)
                        scores[i] -= usage * 0.15

                    top_k = min(15, len(valid_pids))
                    top_indices = np.argsort(scores)[::-1][:top_k]
                    for idx in top_indices:
                        if scores[idx] > -900:  # 有效匹配
                            pid = valid_pids[idx]
                            global_paper_usage[pid] = global_paper_usage.get(pid, 0) + 1
                            target_pids.append(pid)

                # 处理每篇目标文献：检查本地是否存在，否则尝试下载
                for pid in target_pids:
                    meta = paper_db.get(pid, {})
                    # 检查本地路径
                    if meta.get("status") != "local_success":
                        url = meta.get("pdf_url") or meta.get("url")
                        if url and str(url).startswith("http") and not url.lower().endswith(('.jpg', '.png', '.gif')):
                            safe_filename = "paper_" + hashlib.md5(url.encode()).hexdigest()[:8] + ".pdf"
                            local_path = os.path.join(save_dir, safe_filename)

                            if os.path.exists(local_path):
                                meta["local_path"] = local_path
                                meta["status"] = "local_success"
                            else:
                                print(f"  [实时按需下载] 抓取专属文献: {pid}")
                                try:
                                    ctx = ssl.create_default_context()
                                    ctx.check_hostname = False
                                    ctx.verify_mode = ssl.CERT_NONE
                                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                                    with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                                        if 'text/html' not in response.headers.get('Content-Type', '').lower():
                                            with open(local_path, 'wb') as f:
                                                f.write(response.read())
                                            meta["local_path"] = local_path
                                            meta["status"] = "local_success"
                                        else:
                                            meta["status"] = "online_only"
                                except Exception:
                                    meta["status"] = "online_only"

                    # 根据最终状态收集
                    if meta.get("status") == "local_success" and meta.get("local_path"):
                        local_papers.append({"id": pid, "local_path": meta["local_path"]})
                    else:
                        fallback_url = meta.get("pdf_url") or meta.get("url") or pid
                        online_urls.append(fallback_url)

                print(f"  [+] 章节 '{title}' -> 任务就绪: {len(local_papers)} 篇本地 PDF，{len(online_urls)} 条在线摘要。")
                tasks.append({
                    "type": "section",
                    "depth": depth,
                    "title": title,
                    "core_argument": core_arg,
                    "local_papers": local_papers,
                    "online_urls": online_urls
                })
            else:
                # 非叶子节点也可能产生标题任务
                if title or core_arg:
                    tasks.append({"type": "heading", "depth": depth, "title": title, "core_argument": core_arg})
                for k in children_keys:
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
        
        # 🚀【断点续传优化 1：预热缓存】从数据库读取所有历史分块
        # 将 idx 和 title 组合成唯一的主键，防止大纲名字变更导致串台
        existing_records = state.table_reports.all() if hasattr(state, 'table_reports') else []
        cache_db = {f"{r.get('idx')}_{r.get('title')}": r for r in existing_records}
        if cache_db:
            print(f"  [+] 数据库已预载 {len(cache_db)} 条节点记录，随时准备触发断点续传...")

        # 用于在循环中追踪上一段落的内容，强化逻辑衔接
        previous_content = "" 
        
        # 用于追踪当前一级标题的内容，以便在章节末尾生成总结
        current_chapter_title = ""
        current_chapter_content = []
        
        # 用于精确定位大章总结插入位置的指针与数据库索引记录
        current_summary_insert_pos = -1
        current_chapter_start_idx = -1
        
        # 内部函数：为当前一级章节生成总结
        def generate_chapter_summary(summary_idx, summary_title):
            if not current_chapter_title or not current_chapter_content:
                return ""
                
            # 🚀 修复点 1：把 cache_key 设为固定模式，不依赖传入的 summary_title
            summary_cache_key = f"{summary_idx}_summary"
            if summary_cache_key in cache_db and cache_db[summary_cache_key].get("content"):
                print(f"  ⚡ 触发缓存：直接恢复大章小结【{current_chapter_title}】")
                return cache_db[summary_cache_key]["content"]
                
            print(f"  📝 正在为第一级大章【{current_chapter_title}】生成章节总结...")
            summary_prompt = f"""你是一名资深的学术总编。请基于以下提供的本章所有正文内容，写一段精炼的【本章小结】。
            
【章节名称】：{current_chapter_title}
【本章正文内容】：
{"".join(current_chapter_content)}

【写作要求】：
1. 用一段话（大概 150-300 字）总结本章的核心脉络、主要论点和突破。
2. 强调本章各小节之间的内在逻辑联系。
3. 承上启下，为下一章的内容做自然铺垫（如果是最后一章则不需要）。
4. 绝不输出多余的废话，直接输出正文。不需要输出“本章小结”这种小标题。
5. 不要包含引用标号，这是一段纯粹的高层级总结。
"""
            try:
                messages = [{"role": "user", "content": summary_prompt}]
                res = self.writer_agent.model(messages)
                summary_text = getattr(res, 'content', str(res)).strip()
                if summary_text.startswith("```"):
                    summary_text = summary_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
                
                # 🚀 修复点 2：彻底去掉生硬的 **【本章小结】**，让学术段落如丝般顺滑
                final_summary = f"{summary_text}\n\n"
                
                # 写入数据库并缓存
                record = {
                    "idx": summary_idx, 
                    "depth": 1,
                    "title": "", 
                    "content": final_summary, 
                    "core_arg": "本章核心内容总结", 
                    "rag_context": "无", 
                    "local_papers": [], 
                    "online_urls": []
                }
                state.append_report_section(record)
                cache_db[summary_cache_key] = record
                
                return final_summary
                
            except Exception as e:
                print(f"  ⚠️ 生成章节总结失败: {e}")
                return ""

        for idx, task in enumerate(tasks):
            depth = task.get("depth", 1)
            title = task.get("title", "")
            node_type = task.get("type", "heading")
            core_arg = task.get("core_argument", "")

            # --- 在一级标题结束时提取/生成总结，并 insert 填回标题下方 ---
            if depth == 1 and title and title != current_chapter_title:
                if current_chapter_title: 
                    summary_idx = current_chapter_start_idx + 0.01
                    summary_title = f"{current_chapter_title} - 小结"
                    chapter_summary = generate_chapter_summary(summary_idx, summary_title)
                    
                    if chapter_summary and current_summary_insert_pos != -1:
                        report_lines.insert(current_summary_insert_pos, chapter_summary)
                
                # 重置缓存，开始追踪新的第一级章节
                current_chapter_title = title
                current_chapter_content = []

            if title:
                report_lines.append(f"{'#' * (depth+1)} {title}\n\n")

            if node_type == "heading":
                if core_arg:
                    report_lines.append(f"> *{core_arg}*\n\n")
                
                # 捕捉大章标题锚点，如果缓存里没有才写入数据库
                if depth == 1:
                    current_chapter_start_idx = idx
                    current_summary_insert_pos = len(report_lines)
                    
                    cache_key = f"{idx}_{title}"
                    if cache_key not in cache_db:
                        state.append_report_section({
                            "idx": idx,
                            "depth": depth,
                            "title": title,
                            "content": f"> *{core_arg}*\n\n" if core_arg else "",
                            "core_arg": core_arg,
                            "rag_context": "无",
                            "local_papers": [],
                            "online_urls": []
                        })
                        cache_db[cache_key] = True # 标记为已写入
            
            elif node_type == "section":
                
                # ==========================================
                # 🚀【断点续传优化 3：拦截长文写作】直接短路 RAG 和 LLM
                # ==========================================
                cache_key = f"{idx}_{title}"
                if cache_key in cache_db and cache_db[cache_key].get("content"):
                    print(f"  [{idx+1}/{len(tasks)}] ⚡ 触发断点续传：从数据库直接恢复章节 '{title}'")
                    cached_content = cache_db[cache_key]["content"]
                    
                    # 极其重要：同步上下文环境！假装大模型刚刚写完这段，否则后续的小结生成会缺内容
                    previous_content = cached_content
                    current_chapter_content.append(cached_content + "\n")
                    report_lines.append(cached_content + "\n\n")
                    
                    # 跳过后面的 RAG 和 LLM 请求，直接进入下一节
                    continue 
                # ==========================================
                
                # 以下为未命中缓存时，正常发起的大模型检索与撰写逻辑
                local_papers = task.get("local_papers", [])
                online_urls = task.get("online_urls", [])
                
                if local_papers or online_urls:
                    print(f"  [{idx+1}/{len(tasks)}] 🔍 RAG 正在提取 {title} 的语料...")
                    enhanced_query = f"{core_arg} 具体方法 实验数据 提升指标 性能对比 百分比"
                    rag_context = self.rag_tool.forward(query=enhanced_query, local_papers=local_papers, online_urls=online_urls, top_k=8)
                else:
                    rag_context = "无"

                print(f"  [{idx+1}/{len(tasks)}] ✍️ 正在基于SOP框架撰写深度分析初稿...")
                
                prompt = f"""你是一名顶会论文的资深学术主笔。当前正在撰写万字长篇综述《{main_title}》。

【当前所在章节】: {title}
【本节核心论点】: {core_arg}
【前文逻辑承接】(必须确保当前段落的起笔与此平滑衔接):
{previous_content if previous_content else "（本章开篇）"}

【RAG 精准提取语料】:
{rag_context}

【深度融合写作 SOP (标准作业程序)】
你的核心任务是**消除碎片化信息的拼接感**，写出如丝般顺滑的学术正文：
1. 观点先行，文献在后 (Point-First)。
2. 逻辑递进与因果链条：结合【时间演进】与【核心方法对比】，将独立语料串联成逻辑线。
3. 纯净输出：严禁重复章节标题，直接从正文第一句话开始写。所有引用必须严格使用 [REF:xxx] 格式。
"""
                try:
                    res = self.writer_agent.run(prompt)
                    draft_content = getattr(res, 'content', str(res))
                except Exception as e:
                    draft_content = f"撰写失败: {str(e)}"

                print(f"  [{idx+1}/{len(tasks)}] 🔄 正在执行逻辑衔接与润色重写...")
                
                polish_prompt = f"""你是一名极其严苛的《Nature》级别学术总编。请对下面的初稿进行“去机器味”和“去拼接感”的深度重写。

【全局研究主题】：{main_title}
【上一段落结尾】：{previous_content[-150:] if previous_content else "（无）"}
【本段核心论点】：{core_arg}

【待重写初稿】
{draft_content}

【总编重写核心指令】
1. 彻底打碎拼接感：如果初稿像东拼西凑，必须打碎原本生硬的句子结构。
2. 完美咬合前文。
3. 强化论点聚焦。
4. 格式净化：坚决去除 `#`、`**` 等不必要的 Markdown 标题格式。
5. 数据与引用保真：不可删去硬核指标、专有名词和引用标号（如 [REF:p1]）。
"""
                try:
                    messages = [{"role": "user", "content": polish_prompt}]
                    polish_res = self.writer_agent.model(messages)
                    final_content = getattr(polish_res, 'content', str(polish_res)).strip()
                    if final_content.startswith("```"):
                        final_content = final_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
                except Exception as e:
                    print(f"  ⚠️ 润色失败，回退至初稿: {e}")
                    final_content = draft_content

                previous_content = final_content
                current_chapter_content.append(final_content + "\n")
                report_lines.append(final_content + "\n\n")
                
                # 新写完的内容，落盘到数据库并更新当前缓存字典
                record = {
                    "idx": idx, 
                    "depth": depth,
                    "title": title, 
                    "content": final_content, 
                    "core_arg": core_arg, 
                    "rag_context": rag_context, 
                    "local_papers": local_papers, 
                    "online_urls": online_urls
                }
                state.append_report_section(record)
                cache_db[cache_key] = record
                
                print(f"  ✅ 节点完成并归档！")

        # --- 处理最后一章的总结 ---
        if current_chapter_title and current_chapter_content:
            summary_idx = current_chapter_start_idx + 0.01
            summary_title = f"{current_chapter_title} - 小结"
            final_chapter_summary = generate_chapter_summary(summary_idx, summary_title)
            
            if final_chapter_summary and current_summary_insert_pos != -1:
                report_lines.insert(current_summary_insert_pos, final_chapter_summary)

        return "".join(report_lines)
    
class SetVariableTool(Tool):
    name = "tool_set_var"
    description = "将复杂数据存入持久化数据库。"
    inputs = {
        "key": {"type": "string", "description": "变量名"},
        "value": {"type": "any", "description": "任何格式的数据"}
    }
    output_type = "string"

    def forward(self, key: str, value) -> str:
        
        
        # ==========================================
        # 🚀 兼容老架构的“特判拦截路由”
        # ==========================================
        if key == "report_list":
            if isinstance(value, list):
                # 如果大模型试图覆盖写入 report_list，清空旧表并重新插入
                state.table_reports.truncate() 
                for item in value:
                    state.append_report_section(item)
            return "✅ report_list 已无缝映射并同步到 TinyDB 的 reports 专用数据表。"
            
        elif key in ["PAPER_DB", "retrieved_papers"]:
            # 阻止大模型暴力覆盖核心文献库，因为现在有更安全的 upsert 机制
            return f"⚠️ 系统提示：{key} 现已由底层 RAG 引擎自动安全接管入库，不再需要手动覆盖。"

        # ==========================================
        # 常规游离变量的写入
        # ==========================================
        state.set_global_var(key, value)
        return f"✅ 数据已安全落盘至数据库，键名: {key}"

class GetVariableTool(Tool):
    name = "tool_get_var"
    description = "从持久化数据库中读取复杂数据对象。"
    inputs = {"key": {"type": "string", "description": "变量名"}}
    output_type = "any"

    def forward(self, key: str):
        
        
        # ==========================================
        # 🚀 兼容老架构的“欺骗性/组装性”读取
        # 满足 SOP 提示词对老数据结构的依赖，让大模型无感过渡
        # ==========================================
        if key == "report_list":
            # 返回所有报告分块的原始字典列表，并保证顺序正确
            sections = state.table_reports.all()
            sections.sort(key=lambda x: x.get("idx", 0))
            return sections
            
        elif key == "final_report":
            # 动态实时拼接出完整的 Markdown 报告全文
            return state.get_full_report()
            
        elif key == "PAPER_DB":
            # 将列表形式的数据库，转换回老架构 SOP 期望的 {id: paper_data} 字典格式
            all_papers = state.get_all_papers()
            return {p.get("id", p.get("url", str(i))): p for i, p in enumerate(all_papers)}
            
        elif key == "retrieved_papers":
            # 直接返回已爬取的文献列表
            return state.get_all_papers()

        # ==========================================
        # 常规游离变量的读取
        # ==========================================
        val = state.get_global_var(key)
        if val is None:
            return f"❌ 错误：数据库中不存在键名 {key}"
        return val

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
        all_papers = state.get_all_papers()
        # 1. 构建元数据池 (核心修复区)
        meta_lookup = {}
        for p in all_papers:
            pid = p.get("id", "")
            if pid: meta_lookup[pid] = p
            pdf_url = p.get("pdf_url", "")
            if pdf_url: meta_lookup[pdf_url] = p
                    
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

                formatted_content += f"[{num_label}] {' '.join(cite_parts)}\n\n"

        return formatted_content

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

        state.append_report_section({
            "idx": -1, 
            "title": "摘要", 
            "content": formatted_abstract, 
            "core_arg": "全文摘要", 
            "rag_context": "无", 
            "local_papers": [], 
            "online_urls": []
        })

        print("  ✅ 摘要已成功生成并挂载到报告最开头！")
        return state.get_full_report() # 直接从 DB 拉取最新全文


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

        # 🚀 替换点：使用 TinyDB 持久化结论 (设 idx 为 9999 保证在最末尾)
        
        state.append_report_section({
            "idx": 9999, 
            "title": "结论", 
            "content": formatted_conclusion, 
            "core_arg": "全文结论与展望", 
            "rag_context": "无", 
            "local_papers": [], 
            "online_urls": []
        })

        print("  ✅ 结论已成功生成并挂载到报告末尾！")
        return state.get_full_report() # 直接从 DB 拉取最新全文
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
        abs_folder_path = os.path.abspath(folder_path)
            
        if not os.path.exists(abs_folder_path):
            return f"❌ 错误：找不到文件夹绝对路径 {abs_folder_path}"

        pdf_files = [f for f in os.listdir(abs_folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return f"⚠️ 警告：在 {abs_folder_path} 中没有找到任何 PDF 文件。"

        print(f"[LocalInjector] 发现 {len(pdf_files)} 篇本地 PDF，开始提取元数据与精准摘要...")
        
        # 从数据库中拉取已有数据进行去重
        existing_papers = state.get_all_papers()
        existing_titles = {p.get("title", "").strip().lower() for p in existing_papers if p.get("title")}
        existing_original_ids = {p.get("original_id") for p in existing_papers if p.get("original_id")}

        # 🚀 获取本地短 ID 计数器
        max_p_id = 0
        for p in existing_papers:
            pid = p.get("id", "")
            if pid.startswith("p") and pid[1:].isdigit():
                max_p_id = max(max_p_id, int(pid[1:]))
        current_p_id = max_p_id
        
        success_count = 0
        for filename in pdf_files:
            abs_file_path = os.path.join(abs_folder_path, filename)
            
            # 使用原先的 MD5 作为 original_id 防重复扫描
            original_id = "local_" + hashlib.md5(filename.encode()).hexdigest()[:8]
            if original_id in existing_original_ids:
                print(f"  [-] 跳过已存在的本地文献: {filename}")
                continue
            
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

                    # 🚀 修复3：标题去重拦截 (如果这篇本地文献在前面已经被在线API抓过了，直接丢弃本地多余副本)
                    if title.strip().lower() in existing_titles:
                        print(f"  [-] 标题已存在于全局库中，跳过重复文献: {title}")
                        continue

                    # ==========================================
                    # 正则精准提取真实摘要
                    # ==========================================
                    raw_text = ""
                    for page in doc[:2]:
                        raw_text += page.get_text("text") + "\n"
                        
                    abstract_pattern = r'(?i)(?:abstract|摘\s*要)\s*[:\n]?\s*(.*?)(?:\n\s*(?:introduction|引\s*言|1\.\s|keywords|关键\s*词))'
                    match = re.search(abstract_pattern, raw_text, re.DOTALL)
                    
                    if match and len(match.group(1).strip()) > 50:
                        final_abstract = match.group(1).strip()
                        final_abstract = re.sub(r'\n+', ' ', final_abstract)
                    else:
                        final_abstract = raw_text[:1500].strip()
                        final_abstract = re.sub(r'\n+', ' ', final_abstract)
                    current_p_id += 1
                    new_pid = f"p{current_p_id}"
                
            except Exception as e:
                print(f"  [-] 解析 {filename} 失败: {e}")
                continue
                
            # 组装动态获取到的论文元数据
            paper_data = {
                "id": new_pid,
                "original_id": original_id,
                "title": title,
                "abstract": final_abstract if final_abstract else "【无摘要，需依赖RAG正文提取】",
                "local_path": abs_file_path,          
                "status": "local_success",            
                "year": year,                     
                "author": author,                 
                "pdf_url": abs_file_path,             
                "source_query": "local_injection"     
            }
            
            # 🚀 替换点：直接更新到数据库
            state.upsert_paper(paper_data)
            
            existing_original_ids.add(original_id)
            existing_titles.add(title.strip().lower())
            
            success_count += 1
            
        result_msg = f"✅ 成功扫描 {abs_folder_path}，已精准提取 {success_count} 篇本地文献的摘要与元数据！"
        print(f"[LocalInjector] {result_msg}")
        return result_msg