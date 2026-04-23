"""
五步引用验证流程（RAG 增强并发版 - 修复递归与并发死锁、完整版）
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.parse
import os
import random
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

CitationMap = Dict[str, Dict[str, Any]]
StepOutput = Tuple[CitationMap, str]

# ===========================================================================
# 全局并发控制：确保 30 个并发智能体启动时，只初始化一次 ChromaDB 模型
# ===========================================================================
_CHROMA_INIT_LOCK = threading.Lock()
_CHROMA_READY = False


# ===========================================================================
# 1. 模块级引用工具函数
# ===========================================================================

def build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
    authors_text = str(authors or "").strip()
    year_text = str(year or "").strip()
    if authors_text and year_text:
        return f"{authors_text} ({year_text})"
    return str(fallback_title or "").strip()

def normalize_available_citations(raw: Any) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for outer_key, value in raw.items():
            info = dict(value) if isinstance(value, dict) else {"title": str(value or "").strip()}
            key_text = str(outer_key or "").strip()
            title = str(info.get("title") or "").strip() or key_text
            if not title: continue
            info["title"] = title
            authors = str(info.get("authors", "")).strip()
            year = str(info.get("year", "")).strip()
            canonical_key = str(info.get("canonical_key", "")).strip() or build_canonical_citation_key(authors, year, title)
            if canonical_key: info["canonical_key"] = canonical_key
            normalized[title] = info
        return normalized
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict): continue
            info = dict(item)
            title = str(info.get("title", "")).strip() or "Unknown Citation"
            authors = str(info.get("authors", "")).strip()
            year = str(info.get("year", "")).strip()
            canonical_key = build_canonical_citation_key(authors, year, title)
            if canonical_key: info["canonical_key"] = canonical_key
            info["title"] = title
            normalized[title] = info
    return normalized

def build_available_citation_maps(available_citations: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    title_map = normalize_available_citations(available_citations)
    canonical_map: Dict[str, Dict[str, Any]] = {}
    for title, info in title_map.items():
        if not isinstance(info, dict): continue
        row = dict(info)
        row.setdefault("title", title)
        authors = str(row.get("authors", "")).strip()
        year = str(row.get("year", "")).strip()
        canonical_key = str(row.get("canonical_key", "")).strip() or build_canonical_citation_key(authors, year, title)
        if not canonical_key: continue
        row["canonical_key"] = canonical_key
        canonical_map[canonical_key] = row
    return title_map, canonical_map

def enqueue_unverified_citation(agent: Any, author: str, year: str, title: str, claim: str) -> None:
    author_norm = str(author or "").strip()
    year_norm = str(year or "").strip()
    if not author_norm or not year_norm: return
    queue_key = f"{author_norm.lower()}::{year_norm}"
    if queue_key in agent._queued_citation_keys: return
    agent._queued_citation_keys.add(queue_key)
    agent._unverified_citations.append((author_norm, year_norm, str(title or "").strip(), str(claim or "").strip()))

def add_citations(agent: Any, citations: Dict[str, Dict[str, str]], section_title: str = "") -> None:
    title_map, canonical_map = build_available_citation_maps(citations)
    for title, info in title_map.items():
        if title not in agent._citations:
            agent._citation_counter += 1
            agent._citations[title] = info
        else:
            existing = agent._citations.get(title, {})
            existing_title = str(existing.get("title", "")).strip()
            incoming_title = str(info.get("title", "")).strip()
            improved = bool(incoming_title) and incoming_title.lower() not in {"title unavailable", "retrieved from search results"}
            should_update = improved and (not existing_title or existing_title.lower() in {"title unavailable", "retrieved from search results"})
            if should_update:
                existing.update(info)
                agent._citations[title] = existing

    for key, info in canonical_map.items():
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", key)
        author = match.group(1).strip() if match else str(info.get("authors", "")).strip()
        year = match.group(2).strip() if match else str(info.get("year", "")).strip()
        if author and year:
            citation_title = str(info.get("title", "")).strip()
            enqueue_unverified_citation(agent, author, year, citation_title, f"章节「{section_title or '未知章节'}」使用了该引用。")

# ===========================================================================
# 2. 缓存与 RAG 组件
# ===========================================================================

class VerificationCache:
    """持久化缓存：避免对相同标题的文献重复调用 API"""
    def __init__(self, cache_file: str = "./outputs/citation_online_cache.json"):
        self.cache_file = cache_file
        self.cache_data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
            except Exception:
                self.cache_data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception: pass

    def get(self, key: str) -> Optional[Any]: return self.cache_data.get(key)
    def set(self, key: str, value: Any):
        self.cache_data[key] = value
        self.save()

class PaperDownloader:
    """鲁棒的 PDF 下载器：支持重试机制与 DOI 备选解析"""
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'
    ]

    @staticmethod
    def fetch_text(pdf_url: str, abstract: str, doi: str = "", max_retries: int = 2) -> str:
        fallback_text = f"[全文下载失败，降级使用摘要] {abstract}"
        candidates = []
        if pdf_url:
            if "arxiv.org/abs/" in pdf_url: pdf_url = pdf_url.replace("abs", "pdf")
            candidates.append(pdf_url)
        if doi:
            candidates.append(f"https://api.unpaywall.org/v2/{doi}?email=open_deep_research@example.com")
        if not candidates: return fallback_text

        for url in candidates:
            for attempt in range(max_retries):
                try:
                    headers = {'User-Agent': random.choice(PaperDownloader.USER_AGENTS)}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                            continue
                        pdf_bytes = response.read()
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        text = "\n".join([page.get_text() for page in doc])
                        if len(text.strip()) > 500: return text
                except Exception:
                    time.sleep(1)
        return fallback_text

class CitationRAGStore:
    """使用 ChromaDB 管理文献向量，自带并发安全初始化"""
    def __init__(self, persist_dir: str = "./../outputs/chroma_db"):
        global _CHROMA_READY
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # 并发控制：确保第一个到达的实例安全地预热 ChromaDB
        with _CHROMA_INIT_LOCK:
            if not _CHROMA_READY:
                print("🔍 [系统预检] 正在初始化引用验证所需的本地 Embedding 环境...")
                temp_client = chromadb.PersistentClient(path=self.persist_dir)
                temp_client.get_or_create_collection("system-warmup-check")
                _CHROMA_READY = True
                print("✅ [系统预检] 语义模型已就绪，开始并发。")
                
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection("paper_evidence")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.indexed_keys = set()

    def index_paper(self, title_key: str, text: str):
        if not text or title_key in self.indexed_keys: return
        chunks = self.splitter.split_text(text)
        if not chunks: return
        ids = [f"{urllib.parse.quote(title_key)[:50]}_{i}" for i in range(len(chunks))]
        metadatas = [{"title_key": title_key}] * len(chunks)
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        self.indexed_keys.add(title_key)

    def retrieve_evidence_batch(self, claims: List[str], title_key: str, top_k: int = 3) -> str:
        """针对同一篇文献，一次性查询多句话的证据，去重后拼接"""
        if title_key not in self.indexed_keys or not claims: return ""
        try:
            results = self.collection.query(
                query_texts=claims,
                n_results=top_k,
                where={"title_key": title_key}
            )
            unique_evidences = set()
            for doc_list in results.get('documents', []):
                for doc in doc_list: unique_evidences.add(doc)
            return "\n---\n".join(list(unique_evidences))
        except Exception:
            return ""

# ===========================================================================
# 3. 验证器主体
# ===========================================================================

class CitationValidator:
    SOURCE_HINTS = {"arxiv.org": "arxiv", "crossref.org": "crossref", "doi.org": "crossref", "openalex.org": "openalex"}

    def __init__(self, model=None, timeout: int = 5, min_source_count: int = 2, remove_entire_invalid_sentence: bool = True):
        self.model = model
        self.timeout = timeout
        self.min_source_count = max(1, int(min_source_count))
        self.remove_entire_invalid_sentence = remove_entire_invalid_sentence
        
        # 安全初始化外部组件
        self.rag_store = CitationRAGStore()
        self.online_cache = VerificationCache()

    def _verify_title_online(self, title: str) -> Dict[str, Any]:
        if not title or len(title) < 5:
            return {"sources": [], "bibtex": "", "authors": "", "year": "", "url": "", "abstract": "", "pdf_url": "", "doi": ""}

        cache_key = title.strip().lower()
        cached_result = self.online_cache.get(cache_key)
        if cached_result: return cached_result
            
        enrichment = {"sources": [], "bibtex": "", "authors": "", "year": "", "url": "", "abstract": "", "pdf_url": "", "doi": ""}
        encoded_title = urllib.parse.quote(title)
        headers = {'User-Agent': random.choice(PaperDownloader.USER_AGENTS)}
        
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&limit=1&fields=title,authors,year,citationStyles,url,abstract,openAccessPdf,externalIds"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if data.get('data') and len(data['data']) > 0:
                    enrichment["sources"].append("semantic_scholar")
                    paper = data['data'][0]
                    if 'citationStyles' in paper and paper['citationStyles'] and 'bibtex' in paper['citationStyles']:
                        enrichment["bibtex"] = paper['citationStyles']['bibtex']
                    if 'authors' in paper and paper['authors']:
                        enrichment["authors"] = " and ".join([a['name'] for a in paper['authors']])
                    if 'year' in paper and paper['year']: enrichment["year"] = str(paper['year'])
                    if 'url' in paper and paper['url']: enrichment["url"] = paper['url']
                    if 'abstract' in paper and paper['abstract']: enrichment["abstract"] = paper['abstract']
                    if 'openAccessPdf' in paper and paper['openAccessPdf'] and paper['openAccessPdf'].get('url'):
                        enrichment["pdf_url"] = paper['openAccessPdf']['url']
                    if 'externalIds' in paper and paper['externalIds'] and 'DOI' in paper['externalIds']:
                        enrichment["doi"] = paper['externalIds']['DOI']
        except Exception: pass

        if not enrichment["doi"] or not enrichment["abstract"]:
            try:
                url = f"https://api.crossref.org/works?query.bibliographic={encoded_title}&select=DOI,title,author,issued,URL,abstract&rows=1"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    items = data.get('message', {}).get('items', [])
                    if items:
                        enrichment["sources"].append("crossref")
                        item = items[0]
                        doi = item.get("DOI")
                        if doi: enrichment["doi"] = doi
                        if doi and not enrichment["bibtex"]:
                            try:
                                bib_req = urllib.request.Request(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex", headers=headers)
                                with urllib.request.urlopen(bib_req, timeout=3) as bib_resp:
                                    enrichment["bibtex"] = bib_resp.read().decode('utf-8')
                            except Exception: pass
                        if not enrichment["year"] and 'issued' in item:
                            date_parts = item['issued'].get('date-parts', [[]])[0]
                            if date_parts: enrichment["year"] = str(date_parts[0])
                        if not enrichment["authors"] and 'author' in item:
                            enrichment["authors"] = " and ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item['author']])
            except Exception: pass

        self.online_cache.set(cache_key, enrichment)
        return enrichment

    def step1_search(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        updated: CitationMap = {}

        for title_key, item in citations_map.items():
            row = dict(item)
            title = str(row.get("title") or "").strip()
            if not title and row.get("apa_citation"): title = str(row.get("apa_citation")).strip()
            if not title: title = str(title_key).strip()
            
            current_sources = self._normalize_source_list(row.get("source") or row.get("sources"))
            
            if title:
                enrichment = self._verify_title_online(title)
                current_sources = list(set(current_sources + enrichment["sources"]))
                
                if enrichment["bibtex"]:
                    row["standard_bibtex"] = enrichment["bibtex"]
                    row["bibtex"] = enrichment["bibtex"]
                if enrichment["abstract"]: row["abstract"] = enrichment["abstract"]
                if enrichment["pdf_url"]: row["pdf_url"] = enrichment["pdf_url"]
                if enrichment.get("doi"): row["doi"] = enrichment["doi"]
                if enrichment["authors"] and not row.get("authors"): row["authors"] = enrichment["authors"]
                if enrichment["year"] and not row.get("year"): row["year"] = enrichment["year"]
                if enrichment["url"] and not row.get("url"): row["url"] = enrichment["url"]
            
            row["source"] = current_sources
            row["source_count"] = len(current_sources)
            row["step1_status"] = "searched_online"
            updated[str(title_key)] = row

        return updated, self._ensure_paragraph(paragraph)

    def step2_verify(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        updated: CitationMap = {}
        for title_key, item in citations_map.items():
            row = dict(item)
            source_list = self._normalize_source_list(row.get("source"))
            passed = len(source_list) >= self.min_source_count
            row["source"] = source_list
            row["source_count"] = len(source_list)
            row["dual_source_verified"] = passed
            row["verification_status"] = "verified" if passed else "rejected"
            row["verification_counter"] = f"{len(source_list)}/{self.min_source_count}"
            row["step2_status"] = "verified"
            updated[str(title_key)] = row
        return updated, self._ensure_paragraph(paragraph)

    def step3_retrieve(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        updated: CitationMap = {}
        download_tasks = []

        for title_key, item in citations_map.items():
            row = dict(item)
            if not str(row.get("standard_bibtex") or "").strip() and not str(row.get("bibtex") or "").strip():
                row["bibtex"] = self._generate_basic_bibtex(row)
            elif str(row.get("standard_bibtex") or "").strip():
                row["bibtex"] = row["standard_bibtex"]
                
            if row.get("verification_status") == "verified":
                pdf_url = row.get("pdf_url") or row.get("url", "")
                abstract = row.get("abstract", "")
                doi = row.get("doi", "")
                download_tasks.append((title_key, pdf_url, abstract, doi))
            
            row["step3_status"] = "retrieved"
            updated[str(title_key)] = row

        downloaded_texts = {}
        def _fetch(task):
            t_key, p_url, abs_text, doi_val = task
            text = PaperDownloader.fetch_text(p_url, abs_text, doi=doi_val)
            return t_key, text

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {executor.submit(_fetch, task): task for task in download_tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    t_key, text_content = future.result()
                    downloaded_texts[t_key] = text_content
                except Exception: pass

        for t_key, text_content in downloaded_texts.items():
            self.rag_store.index_paper(t_key, text_content)

        return updated, self._ensure_paragraph(paragraph)

    def _call_llm_batch_validation(self, evidence: str, sentences: List[Tuple[int, str]]) -> Dict[int, Tuple[bool, str]]:
        if not self.model or not evidence.strip() or not sentences:
            return {i: (False, "无可用模型或缺乏证据") for i, _ in sentences}

        sentences_str = "\n".join([f"[{idx}]: {sent}" for idx, sent in sentences])
        prompt = f"""
任务：作为学术校验助手，判断提供的论文片段是否充分支持了下列多个目标观点的描述。

论文片段内容（RAG召回证据）：
{evidence}

需要验证的目标句子：
{sentences_str}

要求：仔细核对每一个句子。如果论文片段内容明确支持了该句的核心观点、数据或结论，判定为 true，否则为 false。
你必须严格输出合法的 JSON 格式字典，键为句子的序号，值为判定对象。不要输出任何解释性的废话。
格式示例：
{{
  "0": {{"is_supported": true, "reason": "片段中明确提到了..."}},
  "1": {{"is_supported": false, "reason": "片段中未提及该数据..."}}
}}
"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages) if hasattr(self.model, '__call__') else None
            if not response: return {i: (False, "大模型调用接口不兼容") for i, _ in sentences}
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content).strip()
            
            results_json = json.loads(content)
            parsed_results = {}
            for idx_str, res in results_json.items():
                parsed_results[int(idx_str)] = (bool(res.get("is_supported", False)), str(res.get("reason", "未提供理由")))
            return parsed_results
        except Exception as e:
            return {i: (False, f"模型解析异常: {str(e)}") for i, _ in sentences}

    def step4_validate(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        text = self._ensure_paragraph(paragraph)
        updated: CitationMap = {k: dict(v) for k, v in citations_map.items()}
        
        for row in updated.values():
            row["claim_supported"] = False
            if "validation_result" not in row:
                row["validation_result"] = {"is_valid": False, "reasons": [], "trace_details": []}

        sentences = re.findall(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$", text)
        if not sentences: return updated, text

        citation_to_sents = {}
        for idx, sentence in enumerate(sentences):
            sent = sentence.strip()
            if not sent: continue
            
            # 🔥 修复 1：使用能同时兼容全角（）和半角()的终极正则
            inline_markers = {self._normalize_citation_marker(m) for m in re.findall(r"[\(\（][^\(\)\（\）]{0,220}?(?:19|20)\d{2}[a-z]?[^\(\)\（\）]{0,80}[\)\）]", sent)}
            if not inline_markers: continue

            for title_key, item in updated.items():
                expected_markers = self._build_entry_citation_markers(item)
                matched_markers = expected_markers.intersection(inline_markers)
                title_matched = item.get("title") and len(item["title"]) > 3 and item["title"].lower() in sent.lower()

                if matched_markers or title_matched:
                    if title_key not in citation_to_sents: citation_to_sents[title_key] = []
                    citation_to_sents[title_key].append((idx, sent))

        def _verify_batch_with_evidence(t_key):
            sents_list = citation_to_sents[t_key]
            rag_docs = self.rag_store.retrieve_evidence_batch([s[1] for s in sents_list], t_key, top_k=4)
            if not rag_docs:
                return t_key, rag_docs, {idx: (False, "未能检索到有效全文或摘要片段。") for idx, _ in sents_list}
            return t_key, rag_docs, self._call_llm_batch_validation(rag_docs, sents_list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {executor.submit(_verify_batch_with_evidence, k): k for k in citation_to_sents.keys()}
            for future in concurrent.futures.as_completed(future_to_key):
                t_key, rag_docs, batch_result = future.result()
                item = updated[t_key]
                
                for sent_idx, sent_text in citation_to_sents[t_key]:
                    is_supported, reason = batch_result.get(sent_idx, (False, "LLM未返回此句子的判断"))
                    if is_supported:
                        item["claim_supported"] = True
                        item["validation_result"]["is_valid"] = True
                    
                    item["validation_result"]["trace_details"].append({
                        "sentence": sent_text,
                        "evidence_text": rag_docs,
                        "is_supported": is_supported,
                        "llm_reason": reason
                    })
                    item["validation_result"]["reasons"].append(f"[{'支持' if is_supported else '驳回'}] '{sent_text[:15]}...' -> {reason}")
        
        for row in updated.values():
            if row.get("verification_status") == "verified" and row["claim_supported"]:
                row["verification_status"] = "verified"
            else:
                row["verification_status"] = "rejected"
            row["step4_status"] = "validated"

        return updated, text

    def step5_add(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        text = self._ensure_paragraph(paragraph)

        accepted: CitationMap = {}
        for title_key, item in citations_map.items():
            row = dict(item)
            is_verified = bool(row.get("verification_status") == "verified")
            is_supported = bool(row.get("claim_supported", False))
            if is_verified and is_supported:
                row["step5_status"] = "added"
                accepted[str(title_key)] = row
            else:
                row["step5_status"] = "dropped"

        valid_citation_markers = self._build_valid_citation_markers(accepted)
        cleaned_text = self._remove_invalid_citation_sentences(text, valid_citation_markers)

        return accepted, cleaned_text

    def run_five_step_validation(self, citations: CitationMap, paragraph: str, section_ref: str = "") -> Tuple[CitationMap, str, List[Dict[str, Any]]]:
        steps = [
            (1, "search", self.step1_search),
            (2, "verify", self.step2_verify),
            (3, "retrieve", self.step3_retrieve),
            (4, "validate", self.step4_validate),
            (5, "add", self.step5_add),
        ]

        current_citations, current_paragraph = citations, paragraph
        run_id = f"{int(time.time() * 1000)}"
        validation_logs: List[Dict[str, Any]] = []

        for step_no, action, step_func in steps:
            input_summary = self._build_step_summary(current_citations, current_paragraph)
            started_at = time.time()
            try:
                next_citations, next_paragraph = step_func(current_citations, current_paragraph)
                elapsed_ms = int((time.time() - started_at) * 1000)
                output_summary = self._build_step_summary(next_citations, next_paragraph)
                
                validation_logs.append({
                    "timestamp": time.time(), "run_id": run_id, "section_ref": str(section_ref or ""),
                    "step": step_no, "action": action, "status": "success", "elapsed_ms": elapsed_ms,
                    "input_summary": input_summary, "output_summary": output_summary, "error": "",
                })
                current_citations, current_paragraph = next_citations, next_paragraph
            except Exception as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                validation_logs.append({
                    "timestamp": time.time(), "run_id": run_id, "section_ref": str(section_ref or ""),
                    "step": step_no, "action": action, "status": "failed", "elapsed_ms": elapsed_ms,
                    "input_summary": input_summary, "output_summary": {}, "error": str(exc),
                })
                raise

        return current_citations, current_paragraph, validation_logs

    def _build_step_summary(self, citations: Any, paragraph: Any) -> Dict[str, Any]:
        citation_map = {}
        try: citation_map = self._ensure_citation_map(citations)
        except Exception: pass

        total = len(citation_map)
        verified = sum(1 for item in citation_map.values() if isinstance(item, dict) and str(item.get("verification_status")).strip().lower() == "verified")
        rejected = sum(1 for item in citation_map.values() if isinstance(item, dict) and str(item.get("verification_status")).strip().lower() == "rejected")
        claim_supported = sum(1 for item in citation_map.values() if isinstance(item, dict) and bool(item.get("claim_supported", False)))
        with_source = sum(1 for item in citation_map.values() if isinstance(item, dict) and self._normalize_source_list(item.get("source") or item.get("sources")))
        with_bibtex = sum(1 for item in citation_map.values() if isinstance(item, dict) and str(item.get("standard_bibtex") or item.get("bibtex") or "").strip())

        return {
            "citation_total": total, "verified_count": verified, "rejected_count": rejected,
            "claim_supported_count": claim_supported, "with_source_count": with_source,
            "with_bibtex_count": with_bibtex, "paragraph_length": len(self._ensure_paragraph(paragraph)),
        }

    def _ensure_paragraph(self, paragraph: Any) -> str: return str(paragraph or "").strip()
    
    def _ensure_citation_map(self, citations: Any) -> CitationMap:
        if not isinstance(citations, dict): raise ValueError("citations 必须是 JSON 对象")
        result: CitationMap = {}
        for title, item in citations.items():
            if not isinstance(item, dict): raise ValueError(f"citations[{title!r}] 必须是 JSON 对象")
            result[str(title)] = dict(item)
        return result

    def _normalize_source_list(self, source_value: Any) -> List[str]:
        if source_value is None: return []
        if isinstance(source_value, str): candidates = [s.strip().lower() for s in re.split(r"[,;]", source_value) if s.strip()]
        elif isinstance(source_value, (list, tuple, set)): candidates = [str(s).strip().lower() for s in source_value if str(s).strip()]
        else: raw = str(source_value).strip().lower(); candidates = [raw] if raw else []
        deduped: List[str] = []
        for source in candidates:
            if source and source not in deduped: deduped.append(source)
        return deduped

    def _generate_basic_bibtex(self, row: Dict[str, Any]) -> str:
        title = str(row.get("title") or "Unknown").strip()
        year = str(row.get("year") or "n.d.").strip()
        authors = str(row.get("authors") or "Unknown").strip()
        url = str(row.get("url") or "").strip()
        key_base = re.sub(r"[^a-zA-Z0-9]+", "", title)[:20] or "citation"
        return f"@article{{{key_base}{year},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}},\n  url={{{url}}}\n}}"

    def _build_valid_citation_markers(self, citations: CitationMap) -> set[str]:
        markers: set[str] = set()
        for row in citations.values(): markers.update(self._build_entry_citation_markers(row))
        return markers

    def _build_entry_citation_markers(self, row: Dict[str, Any]) -> set[str]:
        markers: set[str] = set()
        apa = str(row.get("apa_citation") or "").strip()
        if apa: markers.add(self._normalize_citation_marker(apa))
        year = str(row.get("year") or "").strip()
        authors = str(row.get("authors") or "").strip()
        if not year or not authors: return markers
        
        first_author = self._first_author_token(authors)
        if first_author:
            markers.add(self._normalize_citation_marker(f"({first_author}, {year})"))
            markers.add(self._normalize_citation_marker(f"({first_author} et al., {year})"))
            # 🔥 修复 3：兼容大模型爱写的中文“等”
            markers.add(self._normalize_citation_marker(f"({first_author} 等, {year})"))
            
        if "contributors" in authors.lower():
            markers.add(self._normalize_citation_marker(f"({authors}, {year})"))
        return markers

    def _first_author_token(self, authors: str) -> str:
        first_chunk = authors.split(";")[0].strip()
        if not first_chunk: return ""
        if "," in first_chunk: return first_chunk.split(",")[0].strip()
        return first_chunk.strip()

    def _normalize_citation_marker(self, marker: str) -> str:
        text = marker.replace("，", ",").replace("（", "(").replace("）", ")")
        return re.sub(r"\s+", " ", text).strip()

    def _remove_invalid_citation_sentences(self, paragraph: str, valid_markers: set[str]) -> str:
        if not paragraph: return ""
        blocks = [b for b in paragraph.split("\n\n")]
        cleaned_blocks: List[str] = []

        for block in blocks:
            sentences = re.findall(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$", block, flags=re.S)
            kept: List[str] = []
            for sentence in sentences:
                sent = sentence.strip()
                if not sent: continue
                cited_markers = re.findall(r"[\(\（][^\(\)\（\）]{0,220}?(?:19|20)\d{2}[a-z]?[^\(\)\（\）]{0,80}[\)\）]", sent)
                if cited_markers:
                    has_invalid_marker = False
                    for raw_marker in cited_markers:
                        normalized_marker = self._normalize_citation_marker(raw_marker)
                        if normalized_marker not in valid_markers:
                            has_invalid_marker = True
                            if not self.remove_entire_invalid_sentence:
                                sent = sent.replace(raw_marker, "").strip()
                    if has_invalid_marker and self.remove_entire_invalid_sentence:
                        continue 
                if sent: kept.append(sent)

            cleaned_block = "".join(kept).strip()
            if cleaned_block: cleaned_blocks.append(cleaned_block)

        return "\n\n".join(cleaned_blocks).strip()