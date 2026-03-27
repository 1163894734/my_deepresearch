"""
五步引用验证流程（JSON 版）+ 引用工具函数

约束：
- 每一步输入：JSON 格式的引用文献列表 + 当前段落文本
- 每一步输出：修改后的 JSON 引用文献列表 + 修改后的段落文本
- 不兼容旧接口
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

CitationMap = Dict[str, Dict[str, Any]]
StepOutput = Tuple[CitationMap, str]


# ---------------------------------------------------------------------------
# 模块级引用工具函数
# ---------------------------------------------------------------------------

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
            if not title:
                continue
            info["title"] = title
            authors = str(info.get("authors", "")).strip()
            year = str(info.get("year", "")).strip()
            canonical_key = str(info.get("canonical_key", "")).strip() or build_canonical_citation_key(authors, year, title)
            if canonical_key:
                info["canonical_key"] = canonical_key
            normalized[title] = info
        return normalized
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            info = dict(item)
            title = str(info.get("title", "")).strip() or "Unknown Citation"
            authors = str(info.get("authors", "")).strip()
            year = str(info.get("year", "")).strip()
            canonical_key = build_canonical_citation_key(authors, year, title)
            if canonical_key:
                info["canonical_key"] = canonical_key
            info["title"] = title
            normalized[title] = info
    return normalized


def build_available_citation_maps(available_citations: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    title_map = normalize_available_citations(available_citations)
    canonical_map: Dict[str, Dict[str, Any]] = {}
    for title, info in title_map.items():
        if not isinstance(info, dict):
            continue
        row = dict(info)
        row.setdefault("title", title)
        authors = str(row.get("authors", "")).strip()
        year = str(row.get("year", "")).strip()
        canonical_key = str(row.get("canonical_key", "")).strip() or build_canonical_citation_key(authors, year, title)
        if not canonical_key:
            continue
        row["canonical_key"] = canonical_key
        canonical_map[canonical_key] = row
    return title_map, canonical_map


def enqueue_unverified_citation(agent: Any, author: str, year: str, title: str, claim: str) -> None:
    author_norm = str(author or "").strip()
    year_norm = str(year or "").strip()
    if not author_norm or not year_norm:
        return
    queue_key = f"{author_norm.lower()}::{year_norm}"
    if queue_key in agent._queued_citation_keys:
        return
    agent._queued_citation_keys.add(queue_key)
    agent._unverified_citations.append((author_norm, year_norm, str(title or "").strip(), str(claim or "").strip()))


def add_citations(agent: Any, citations: Dict[str, Dict[str, str]], section_title: str = "") -> None:
    # 同时获取 title_map 和 canonical_map
    title_map, canonical_map = build_available_citation_maps(citations)
    
    # 1. 【核心修复】：统一使用 title 作为唯一的 Key 存入字典，彻底告别分身！
    for title, info in title_map.items():
        if title not in agent._citations:
            agent._citation_counter += 1
            agent._citations[title] = info
        else:
            # 如果标题已存在，按需合并更新更丰富的信息
            existing = agent._citations.get(title, {})
            existing_title = str(existing.get("title", "")).strip()
            incoming_title = str(info.get("title", "")).strip()
            improved = bool(incoming_title) and incoming_title.lower() not in {
                "title unavailable", "retrieved from search results"
            }
            should_update = improved and (not existing_title or existing_title.lower() in {
                "title unavailable", "retrieved from search results"
            })
            if should_update:
                existing.update(info)
                agent._citations[title] = existing

    # 2. 保留底层的日志追踪功能（仅发往 reference_trace_log，不污染字典）
    for key, info in canonical_map.items():
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", key)
        author = match.group(1).strip() if match else str(info.get("authors", "")).strip()
        year = match.group(2).strip() if match else str(info.get("year", "")).strip()
        
        if author and year:
            citation_title = str(info.get("title", "")).strip()
            enqueue_unverified_citation(agent, author, year, citation_title, f"章节「{section_title or '未知章节'}」使用了该引用。")


class CitationValidator:
    """自带在线双源验证、标准BibTeX抓取能力与宽容匹配规则的引用验证器。"""

    SOURCE_HINTS = {
        "arxiv.org": "arxiv",
        "crossref.org": "crossref",
        "doi.org": "crossref",
        "openalex.org": "openalex",
    }

    def __init__(self, model=None, timeout: int = 5, min_source_count: int = 2, remove_entire_invalid_sentence: bool = True):
        self.model = model
        self.timeout = timeout
        self.min_source_count = max(1, int(min_source_count))
        
        # 【新增配置】：处理不合规引用时，True表示删除整句话，False表示只抠掉(Author, Year)括号
        self.remove_entire_invalid_sentence = remove_entire_invalid_sentence

    def _verify_title_online(self, title: str) -> Dict[str, Any]:
        """在线查证工具：返回命中的平台列表、标准BibTeX以及核心元数据补充"""
        enrichment = {
            "sources": [],
            "bibtex": "",
            "authors": "",
            "year": "",
            "url": ""
        }
        if not title or len(title) < 5:
            return enrichment
            
        encoded_title = urllib.parse.quote(title)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # 1. 查 Semantic Scholar (它能最直接提供完美的 BibTeX)
        try:
            # 增加 citationStyles 字段，直接要求返回 BibTeX
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&limit=1&fields=title,authors,year,citationStyles,url"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get('data') and len(data['data']) > 0:
                    enrichment["sources"].append("semantic_scholar")
                    paper = data['data'][0]
                    
                    # 抓取标准 BibTeX
                    if 'citationStyles' in paper and paper['citationStyles'] and 'bibtex' in paper['citationStyles']:
                        enrichment["bibtex"] = paper['citationStyles']['bibtex']
                        
                    if 'authors' in paper and paper['authors']:
                        enrichment["authors"] = " and ".join([a['name'] for a in paper['authors']])
                    if 'year' in paper and paper['year']:
                        enrichment["year"] = str(paper['year'])
                    if 'url' in paper and paper['url']:
                        enrichment["url"] = paper['url']
        except Exception:
            pass

        # 2. 查 Crossref (DOI核心库，若 S2 没给 BibTeX，可通过 DOI 专门请求)
        try:
            url = f"https://api.crossref.org/works?query.bibliographic={encoded_title}&select=DOI,title,author,issued,URL&rows=1"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                items = data.get('message', {}).get('items', [])
                if items:
                    enrichment["sources"].append("crossref")
                    item = items[0]
                    doi = item.get("DOI")
                    
                    # 若依然没拿到 BibTeX，通过 Crossref 提供的 transform 接口抓取
                    if doi and not enrichment["bibtex"]:
                        try:
                            bib_req = urllib.request.Request(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex", headers=headers)
                            with urllib.request.urlopen(bib_req, timeout=3) as bib_resp:
                                enrichment["bibtex"] = bib_resp.read().decode('utf-8')
                        except Exception:
                            pass
                    
                    if not enrichment["year"] and 'issued' in item:
                        date_parts = item['issued'].get('date-parts', [[]])[0]
                        if date_parts:
                            enrichment["year"] = str(date_parts[0])
                    if not enrichment["authors"] and 'author' in item:
                        enrichment["authors"] = " and ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item['author']])
                    if not enrichment["url"] and 'URL' in item:
                        enrichment["url"] = item['URL']
        except Exception:
            pass

        # 3. 查 ArXiv (作为双源的辅助确认)
        try:
            url = f"http://export.arxiv.org/api/query?search_query=ti:%22{encoded_title}%22&max_results=1"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = resp.read().decode()
                if "<entry>" in data:
                    enrichment["sources"].append("arxiv")
                    if not enrichment["year"]:
                        year_match = re.search(r'<published>(\d{4})', data)
                        if year_match: enrichment["year"] = year_match.group(1)
                    if not enrichment["authors"]:
                        authors = re.findall(r'<name>\s*([^<]+)\s*</name>', data)
                        if authors: enrichment["authors"] = " and ".join(authors)
                    if not enrichment["url"]:
                        id_match = re.search(r'<id>(.*?)</id>', data)
                        if id_match: enrichment["url"] = id_match.group(1)
        except Exception:
            pass

        return enrichment

    def step1_search(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        updated: CitationMap = {}

        for title_key, item in citations_map.items():
            row = dict(item)
            
            title = str(row.get("title") or "").strip()
            if not title and row.get("apa_citation"):
                title = str(row.get("apa_citation")).strip()
            if not title:
                title = str(title_key).strip()
            
            current_sources = self._normalize_source_list(row.get("source") or row.get("sources"))
            
            # 在线核实并获取增强数据（包括标准 BibTeX）
            if title:
                enrichment = self._verify_title_online(title)
                
                # 合并源头并去重
                current_sources = list(set(current_sources + enrichment["sources"]))
                
                # 新增核心：将标准 BibTeX 存入字典
                if enrichment["bibtex"]:
                    row["standard_bibtex"] = enrichment["bibtex"]
                    row["bibtex"] = enrichment["bibtex"]  # 顺便覆盖通用的 bibtex 字段
                
                # 动态补充缺失的元数据
                if enrichment["authors"] and not row.get("authors"):
                    row["authors"] = enrichment["authors"]
                if enrichment["year"] and not row.get("year"):
                    row["year"] = enrichment["year"]
                if enrichment["url"] and not row.get("url"):
                    row["url"] = enrichment["url"]
            
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
        for title_key, item in citations_map.items():
            row = dict(item)
            
            # 改进：如果我们通过在线搜索拿到了 standard_bibtex，则不要用粗糙生成器覆盖它
            if not str(row.get("standard_bibtex") or "").strip() and not str(row.get("bibtex") or "").strip():
                row["bibtex"] = self._generate_basic_bibtex(row)
            elif str(row.get("standard_bibtex") or "").strip():
                row["bibtex"] = row["standard_bibtex"]
                
            row["step3_status"] = "retrieved"
            updated[str(title_key)] = row
        return updated, self._ensure_paragraph(paragraph)

    def step4_validate(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        text = self._ensure_paragraph(paragraph)
        
        inline_markers_in_text = {
            self._normalize_citation_marker(m)
            for m in re.findall(r"\([^()]*\d{4}[a-z]?[^()]*\)", text)
        }

        updated: CitationMap = {}
        for title_key, item in citations_map.items():
            row = dict(item)
            
            # 1. 标题直接匹配测试
            title = str(row.get("title", "")).strip()
            title_matched = False
            if title and len(title) > 3 and title.lower() in text.lower():
                title_matched = True
            
            # 2. APA严格正则匹配测试
            expected_markers = self._build_entry_citation_markers(row)
            matched_markers = sorted(expected_markers.intersection(inline_markers_in_text))
            
            # 3. 作者+年份宽容组合测试
            authors_text = str(row.get("authors") or "").lower()
            year_text = str(row.get("year") or "").strip()
            loose_matched = False
            if year_text and year_text in text:
                first_name = re.split(r'[,;\s]', authors_text)[0].strip()
                if first_name and first_name in text.lower():
                    loose_matched = True

            claim_supported = bool(matched_markers) or title_matched or loose_matched
            confidence = 1.0 if claim_supported else 0.0

            row["validation_result"] = {
                "is_valid": claim_supported,
                "confidence": confidence,
                "reason": "title-or-apa-or-loose-match",
                "matched_markers": matched_markers,
            }
            row["claim_supported"] = claim_supported
            
            if row.get("verification_status") == "verified" and claim_supported:
                row["verification_status"] = "verified"
            else:
                row["verification_status"] = "rejected"
                
            row["step4_status"] = "validated"
            updated[str(title_key)] = row

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

        # 拿到清洗完无效引用的纯净正文
        cleaned_text = self._remove_invalid_citation_sentences(text, valid_citation_markers)

        # 【核心修改】：直接使用纯净文本，不再拼接 [Validated References] 尾巴
        updated_paragraph = cleaned_text

        return accepted, updated_paragraph

    def run_five_step_validation(
        self,
        citations: CitationMap,
        paragraph: str,
        log_file_path: Optional[str] = None,
        section_ref: str = "",
    ) -> StepOutput:
        steps = [
            (1, "search", self.step1_search),
            (2, "verify", self.step2_verify),
            (3, "retrieve", self.step3_retrieve),
            (4, "validate", self.step4_validate),
            (5, "add", self.step5_add),
        ]

        current_citations, current_paragraph = citations, paragraph
        run_id = f"{int(time.time() * 1000)}"

        for step_no, action, step_func in steps:
            input_summary = self._build_step_summary(current_citations, current_paragraph)
            started_at = time.time()
            try:
                next_citations, next_paragraph = step_func(current_citations, current_paragraph)
                elapsed_ms = int((time.time() - started_at) * 1000)
                output_summary = self._build_step_summary(next_citations, next_paragraph)
                self._append_five_step_log(
                    log_file_path=log_file_path,
                    run_id=run_id,
                    section_ref=section_ref,
                    step_no=step_no,
                    action=action,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    elapsed_ms=elapsed_ms,
                    status="success",
                )
                current_citations, current_paragraph = next_citations, next_paragraph
            except Exception as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                self._append_five_step_log(
                    log_file_path=log_file_path,
                    run_id=run_id,
                    section_ref=section_ref,
                    step_no=step_no,
                    action=action,
                    input_summary=input_summary,
                    output_summary={},
                    elapsed_ms=elapsed_ms,
                    status="failed",
                    error=str(exc),
                )
                raise

        return current_citations, current_paragraph

    def _build_step_summary(self, citations: Any, paragraph: Any) -> Dict[str, Any]:
        citation_map: CitationMap = {}
        try:
            citation_map = self._ensure_citation_map(citations)
        except Exception:
            citation_map = {}

        total = len(citation_map)
        verified = 0
        rejected = 0
        claim_supported = 0
        with_source = 0
        with_bibtex = 0

        for item in citation_map.values():
            if not isinstance(item, dict):
                continue
            if str(item.get("verification_status") or "").strip().lower() == "verified":
                verified += 1
            if str(item.get("verification_status") or "").strip().lower() == "rejected":
                rejected += 1
            if bool(item.get("claim_supported", False)):
                claim_supported += 1
            if self._normalize_source_list(item.get("source") or item.get("sources")):
                with_source += 1
            if str(item.get("standard_bibtex") or item.get("bibtex") or "").strip():
                with_bibtex += 1

        paragraph_text = self._ensure_paragraph(paragraph)
        return {
            "citation_total": total,
            "verified_count": verified,
            "rejected_count": rejected,
            "claim_supported_count": claim_supported,
            "with_source_count": with_source,
            "with_bibtex_count": with_bibtex,
            "paragraph_length": len(paragraph_text),
        }

    def _append_five_step_log(
        self,
        log_file_path: Optional[str],
        run_id: str,
        section_ref: str,
        step_no: int,
        action: str,
        input_summary: Dict[str, Any],
        output_summary: Dict[str, Any],
        elapsed_ms: int,
        status: str,
        error: str = "",
    ) -> None:
        if not log_file_path:
            return

        try:
            payload: Dict[str, Any] = {}
            try:
                with open(log_file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}

            logs = payload.get("five_step_logs", [])
            if not isinstance(logs, list):
                logs = []

            logs.append(
                {
                    "timestamp": time.time(),
                    "run_id": run_id,
                    "section_ref": str(section_ref or ""),
                    "step": step_no,
                    "action": action,
                    "status": status,
                    "elapsed_ms": int(elapsed_ms),
                    "input_summary": input_summary,
                    "output_summary": output_summary,
                    "error": str(error or ""),
                }
            )

            payload["five_step_logs"] = logs
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            return

    def _ensure_paragraph(self, paragraph: Any) -> str:
        return str(paragraph or "").strip()

    def _ensure_citation_map(self, citations: Any) -> CitationMap:
        if not isinstance(citations, dict):
            raise ValueError("citations 必须是 JSON 对象（Dict[title, citation_info]）")

        result: CitationMap = {}
        for title, item in citations.items():
            if not isinstance(item, dict):
                raise ValueError(f"citations[{title!r}] 必须是 JSON 对象（Dict）")
            result[str(title)] = dict(item)
        return result

    def _normalize_source_list(self, source_value: Any) -> List[str]:
        if source_value is None:
            return []

        if isinstance(source_value, str):
            candidates = [s.strip().lower() for s in re.split(r"[,;]", source_value) if s.strip()]
        elif isinstance(source_value, (list, tuple, set)):
            candidates = [str(s).strip().lower() for s in source_value if str(s).strip()]
        else:
            raw = str(source_value).strip().lower()
            candidates = [raw] if raw else []

        deduped: List[str] = []
        for source in candidates:
            if source and source not in deduped:
                deduped.append(source)
        return deduped

    def _source_from_url(self, url: str) -> str:
        low_url = url.lower()
        for host, source in self.SOURCE_HINTS.items():
            if host in low_url:
                return source
        return ""

    def _title_tokens(self, title: str) -> List[str]:
        raw_tokens = re.findall(r"[a-zA-Z]{4,}", title.lower())
        stop_words = {
            "from", "with", "into", "that", "this", "have", "been", "their",
            "using", "large", "model", "models", "language",
        }
        tokens = [t for t in raw_tokens if t not in stop_words]
        deduped: List[str] = []
        for token in tokens:
            if token not in deduped:
                deduped.append(token)
        return deduped[:8]

    def _generate_basic_bibtex(self, row: Dict[str, Any]) -> str:
        title = str(row.get("title") or "Unknown").strip()
        year = str(row.get("year") or "n.d.").strip()
        authors = str(row.get("authors") or "Unknown").strip()
        url = str(row.get("url") or "").strip()

        key_base = re.sub(r"[^a-zA-Z0-9]+", "", title)[:20] or "citation"
        key = f"{key_base}{year}"

        return (
            f"@article{{{key},\n"
            f"  title={{{title}}},\n"
            f"  author={{{authors}}},\n"
            f"  year={{{year}}},\n"
            f"  url={{{url}}}\n"
            f"}}"
        )

    def _build_valid_citation_markers(self, citations: CitationMap) -> set[str]:
        markers: set[str] = set()
        for row in citations.values():
            markers.update(self._build_entry_citation_markers(row))
        return markers

    def _build_entry_citation_markers(self, row: Dict[str, Any]) -> set[str]:
        markers: set[str] = set()
        apa = str(row.get("apa_citation") or "").strip()
        if apa:
            markers.add(self._normalize_citation_marker(apa))

        year = str(row.get("year") or "").strip()
        authors = str(row.get("authors") or "").strip()
        if not year or not authors:
            return markers

        first_author = self._first_author_token(authors)
        if first_author:
            markers.add(self._normalize_citation_marker(f"({first_author}, {year})"))
            markers.add(self._normalize_citation_marker(f"({first_author} et al., {year})"))

        if "contributors" in authors.lower():
            markers.add(self._normalize_citation_marker(f"({authors}, {year})"))

        return markers

    def _first_author_token(self, authors: str) -> str:
        first_chunk = authors.split(";")[0].strip()
        if not first_chunk:
            return ""
        if "," in first_chunk:
            return first_chunk.split(",")[0].strip()
        return first_chunk.strip()

    def _normalize_citation_marker(self, marker: str) -> str:
        text = marker.replace("，", ",")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _remove_invalid_citation_sentences(self, paragraph: str, valid_markers: set[str]) -> str:
        """
        根据配置，清理正文中的非法引用。
        - 若 remove_entire_invalid_sentence=True: 只要句子里含有一个非法引用，整句话直接删掉。
        - 若 remove_entire_invalid_sentence=False: 只抹除句子里非法的 "(Author, Year)" 标记，保留正文内容。
        """
        if not paragraph:
            return ""

        blocks = [b for b in paragraph.split("\n\n")]
        cleaned_blocks: List[str] = []

        for block in blocks:
            # 使用正则按中文/英文句号、问号、叹号进行安全分句，保留标点符号
            sentences = re.findall(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$", block, flags=re.S)
            kept: List[str] = []
            
            for sentence in sentences:
                sent = sentence.strip()
                if not sent:
                    continue

                # 提取当前句子中所有的文献引用标记 (匹配典型的 APA 格式)
                cited_markers = re.findall(
                    r"[\(\（][^\(\)\（\）]{0,220}?(?:19|20)\d{2}[a-z]?[^\(\)\（\）]{0,80}[\)\）]",
                    sent,
                )
                
                if cited_markers:
                    has_invalid_marker = False
                    
                    for raw_marker in cited_markers:
                        normalized_marker = self._normalize_citation_marker(raw_marker)
                        
                        # 如果发现这个标记不在合法名单 (valid_markers) 里
                        if normalized_marker not in valid_markers:
                            has_invalid_marker = True
                            
                            # 如果配置为“只删括号”，就在这里把非法的括号抠掉
                            if not self.remove_entire_invalid_sentence:
                                sent = sent.replace(raw_marker, "").strip()
                    
                    # 如果句子里存在非法引用，且配置为“删除整句”，则直接丢弃这句话
                    if has_invalid_marker and self.remove_entire_invalid_sentence:
                        continue 
                
                # 如果句子存活下来了（或者被抠掉括号后还有内容），加回最终段落
                if sent:
                    kept.append(sent)

            # 把幸存的句子重新拼成段落
            cleaned_block = "".join(kept).strip()
            if cleaned_block:
                cleaned_blocks.append(cleaned_block)

        return "\n\n".join(cleaned_blocks).strip()