"""
五步引用验证流程（JSON 版）+ 引用工具函数

约束：
- 每一步输入：JSON 格式的引用文献列表 + 当前段落文本
- 每一步输出：修改后的 JSON 引用文献列表 + 修改后的段落文本
- 不兼容旧接口
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


CitationMap = Dict[str, Dict[str, Any]]
StepOutput = Tuple[CitationMap, str]


# ---------------------------------------------------------------------------
# 模块级引用工具函数（原 CitationFlowService，不含 LLM rewrite）
# ---------------------------------------------------------------------------

def build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
    authors_text = str(authors or "").strip()
    year_text = str(year or "").strip()
    if authors_text and year_text:
        return f"{authors_text} ({year_text})"
    return str(fallback_title or "").strip()


def extract_author_year_from_key(key: str) -> Tuple[str, str]:
    text = str(key or "").strip()
    match = re.match(r"^(.+?)\s*\(([^)]+)\)$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""


def get_first_author_surface(authors: str) -> str:
    raw = str(authors or "").strip()
    if not raw:
        return ""
    first = re.split(r"\s*(?:;|,|&| and )\s*", raw, maxsplit=1)[0].strip() or raw
    parts = [p for p in re.split(r"\s+", first) if p]
    if re.search(r"[A-Za-z]", first) and len(parts) > 1:
        return parts[-1].strip(".,")
    return first.strip(".,")


def build_preferred_inline_citation(authors: str, year: str) -> str:
    authors_text = str(authors or "").strip()
    year_text = str(year or "").strip()
    if not authors_text or not year_text:
        return ""
    if "et al." in authors_text.lower():
        label = authors_text
    else:
        multi_author = bool(re.search(r"[,;&]", authors_text) or " and " in authors_text.lower())
        first_author = get_first_author_surface(authors_text)
        label = f"{first_author} et al." if (multi_author and first_author) else (first_author or authors_text)
    return f"({label}, {year_text})"


def normalize_citation_surface(citation_text: str) -> str:
    text = re.sub(r"\s+", " ", str(citation_text or "").strip())
    if not text:
        return ""
    parenthetical = re.match(r"^\((.+?),\s*((?:19|20)\d{2}|n\.d\.)\)$", text)
    if parenthetical:
        return f"{parenthetical.group(1).strip()} ({parenthetical.group(2).strip()})"
    narrative = re.match(r"^(.+?)\s*\(((?:19|20)\d{2}|n\.d\.)\)$", text)
    if narrative:
        return f"{narrative.group(1).strip()} ({narrative.group(2).strip()})"
    return text


def build_allowed_citation_alias_map(allowed_citation_keys: Optional[Set[str]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not allowed_citation_keys:
        return alias_map
    for canonical_key in allowed_citation_keys:
        canonical = str(canonical_key or "").strip()
        if not canonical:
            continue
        authors, year = extract_author_year_from_key(canonical)
        if not authors or not year:
            continue
        variants: Set[str] = {canonical, f"({authors}, {year})"}
        first_author = get_first_author_surface(authors)
        if first_author:
            variants.add(f"{first_author} ({year})")
            variants.add(f"({first_author}, {year})")
        multi_author = bool(re.search(r"[,;&]", authors) or " and " in authors.lower())
        if multi_author and first_author:
            etal = f"{first_author} et al."
            variants.add(f"{etal} ({year})")
            variants.add(f"({etal}, {year})")
        for variant in variants:
            normalized = normalize_citation_surface(variant)
            if normalized:
                alias_map[normalized] = canonical
    return alias_map


def resolve_allowed_citation_key(raw_key: str, allowed_citation_keys: Optional[Set[str]]) -> Optional[str]:
    if not allowed_citation_keys:
        return None
    alias_map = build_allowed_citation_alias_map(allowed_citation_keys)
    return alias_map.get(normalize_citation_surface(raw_key))


def scan_citations_in_text(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    seen_spans: Set[Tuple[int, int]] = set()
    content = str(text or "")
    parenthetical_pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
    narrative_pattern = (
        r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-.]*"
        r"(?:\s+(?:et al\.|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-.]*|AI|Research|Institute|Team|Face|Insights|Newsroom))*)"
        r"\s*\(((?:19|20)\d{2}|n\.d\.)\)"
    )
    for match in re.finditer(parenthetical_pattern, content):
        start, end = match.span()
        seen_spans.add((start, end))
        findings.append({
            "kind": "parenthetical",
            "raw_key": f"{match.group(1).strip()} ({match.group(2).strip()})",
            "matched_text": match.group(0),
            "start": start,
            "end": end,
        })
    for match in re.finditer(narrative_pattern, content):
        start, end = match.span()
        if (start, end) in seen_spans:
            continue
        findings.append({
            "kind": "narrative",
            "raw_key": f"{match.group(1).strip()} ({match.group(2).strip()})",
            "matched_text": match.group(0),
            "start": start,
            "end": end,
        })
    findings.sort(key=lambda item: item.get("start", 0))
    return findings


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


def format_allowed_citation_whitelist(available_citations: Dict[str, Dict[str, str]]) -> str:
    if not available_citations:
        return "（当前段落没有可用引用，禁止输出任何 APA 文内引用）"
    _, canonical_map = build_available_citation_maps(available_citations)
    lines = []
    for idx, (key, info) in enumerate(canonical_map.items(), 1):
        authors = str(info.get("authors") or extract_author_year_from_key(key)[0]).strip()
        year = str(info.get("year") or extract_author_year_from_key(key)[1]).strip()
        title = str(info.get("title", "")).strip() or "Untitled source"
        inline = str(info.get("inline_citation", "")).strip() or build_preferred_inline_citation(authors, year)
        lines.append(f"- [{idx}] 只允许使用: {inline} | canonical_key: {key} | title: {title}")
    return "\n".join(lines)


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
    _, canonical_map = build_available_citation_maps(citations)
    for key, info in canonical_map.items():
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", key)
        author = match.group(1).strip() if match else str(info.get("authors", "")).strip()
        year = match.group(2).strip() if match else str(info.get("year", "")).strip()
        if key not in agent._citations:
            agent._citation_counter += 1
            agent._citations[key] = info
        else:
            existing = agent._citations.get(key, {})
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
                agent._citations[key] = existing
        if author and year:
            citation_title = str(agent._citations.get(key, {}).get("title") or info.get("title") or "").strip()
            enqueue_unverified_citation(agent, author, year, citation_title, f"章节「{section_title or '未知章节'}」使用了该引用。")


def collect_citations_from_text(
    agent: Any,
    text: str,
    section_title: str = "",
    allowed_citation_keys: Optional[Set[str]] = None,
    add_new_from_text: bool = False,
) -> None:
    if not text:
        return
    normalized_allowed_keys: Optional[Set[str]] = None
    if allowed_citation_keys is not None:
        normalized_allowed_keys = {str(k).strip() for k in allowed_citation_keys if str(k).strip()}
    pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
    placeholder_authors = {"张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"}
    sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])\s+|\n+', str(text)) if s.strip()]
    for sentence in sentences:
        for author_raw, year_raw in re.findall(pattern, sentence):
            author = author_raw.strip()
            year = year_raw.strip()
            if author in placeholder_authors or author.lower().replace(" ", "") in placeholder_authors:
                continue
            if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
                continue
            key = f"{author} ({year})"
            resolved_key = resolve_allowed_citation_key(key, normalized_allowed_keys)
            if normalized_allowed_keys is not None and resolved_key is None:
                continue
            if resolved_key:
                key = resolved_key
                author, year = extract_author_year_from_key(resolved_key)
            if key not in agent._citations:
                if add_new_from_text:
                    agent._citation_counter += 1
                    agent._citations[key] = {
                        "id": agent._citation_counter,
                        "authors": author,
                        "year": year,
                        "title": "Title unavailable",
                        "title_source": "generated_text_only",
                        "title_note": "仅从正文 APA 文内引用提取，原句不含文献题名",
                    }
                else:
                    continue
            enqueue_unverified_citation(
                agent,
                author=author,
                year=year,
                title=str(agent._citations.get(key, {}).get("title") or "").strip(),
                claim=sentence,
            )


def extract_citations_from_rag(agent: Any, rag_context: str) -> Dict[str, Dict[str, str]]:
    citations: Dict[str, Dict[str, str]] = {}
    latest_payload = agent.state.get(agent.STATE_LATEST_TAGGED_SEARCH_PAYLOAD, {})
    if isinstance(latest_payload, dict):
        citations.update(agent._tagged_search_service.extract_citations_from_tagged_payload(agent, latest_payload))
    if not rag_context or agent.strict_citation_flow:
        return citations
    placeholder_authors = {"张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"}
    for author_raw, year_raw in re.findall(r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)", str(rag_context)):
        author = author_raw.strip()
        year = year_raw.strip()
        if author in placeholder_authors or author.lower().replace(" ", "") in placeholder_authors:
            continue
        if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
            continue
        canonical_key = f"{author} ({year})"
        title_key = f"Retrieved from search results - {canonical_key}"
        if title_key not in citations:
            citations[title_key] = {
                "authors": author,
                "year": year,
                "title": "Retrieved from search results",
                "title_source": "rag_context",
                "canonical_key": canonical_key,
            }
    return citations


class CitationValidator:
    """仅支持 JSON 引用列表 + 段落文本的五步验证器。"""

    SOURCE_HINTS = {
        "arxiv.org": "arxiv",
        "crossref.org": "crossref",
        "doi.org": "crossref",
        "openalex.org": "openalex",
    }

    def __init__(self, model=None, timeout: int = 5, min_source_count: int = 2):
        self.model = model
        self.timeout = timeout
        self.min_source_count = max(1, int(min_source_count))

    def step1_search(self, citations: CitationMap, paragraph: str) -> StepOutput:
        citations_map = self._ensure_citation_map(citations)
        updated: CitationMap = {}
        for title_key, item in citations_map.items():
            row = dict(item)
            if not str(row.get("title") or "").strip():
                row["title"] = str(title_key)
            source_list = self._normalize_source_list(row.get("source") or row.get("sources"))
            url_source = self._source_from_url(str(row.get("url") or ""))
            if url_source and url_source not in source_list:
                source_list.append(url_source)
            row["source"] = source_list
            row["source_count"] = len(source_list)
            row["step1_status"] = "searched"
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
            if not str(row.get("bibtex") or "").strip():
                row["bibtex"] = self._generate_basic_bibtex(row)
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
            expected_markers = self._build_entry_citation_markers(row)
            matched_markers = sorted(expected_markers.intersection(inline_markers_in_text))
            claim_supported = bool(matched_markers)
            confidence = 1.0 if claim_supported else 0.0

            row["validation_result"] = {
                "is_valid": claim_supported,
                "confidence": confidence,
                "reason": "apa-inline-citation-match",
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

        if accepted:
            titles = "；".join(str(c.get("title") or "Unknown") for c in accepted.values())
            updated_paragraph = f"{cleaned_text}\n\n[Validated References] {titles}" if cleaned_text else f"[Validated References] {titles}"
        else:
            updated_paragraph = cleaned_text

        return accepted, updated_paragraph

    def run_five_step_validation(self, citations: CitationMap, paragraph: str) -> StepOutput:
        c1, p1 = self.step1_search(citations, paragraph)
        c2, p2 = self.step2_verify(c1, p1)
        c3, p3 = self.step3_retrieve(c2, p2)
        c4, p4 = self.step4_validate(c3, p3)
        c5, p5 = self.step5_add(c4, p4)
        return c5, p5

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
        if not paragraph:
            return ""

        blocks = [b for b in paragraph.split("\n\n")]
        cleaned_blocks: List[str] = []

        for block in blocks:
            # 不按英文句点 "." 切句，避免把 "et al., 2024" 这类引用拆断
            sentences = re.findall(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$", block, flags=re.S)
            kept: List[str] = []
            for sentence in sentences:
                sent = sentence.strip()
                if not sent:
                    continue

                cited_markers = re.findall(
                    r"[\(\（][^\(\)\（\）]{0,220}?(?:19|20)\d{2}[a-zA-Z]?[^\(\)\（\）]{0,80}[\)\）]",
                    sent,
                )
                if not cited_markers:
                    kept.append(sent)
                    continue

                normalized = [self._normalize_citation_marker(c) for c in cited_markers]
                if all(c in valid_markers for c in normalized):
                    kept.append(sent)

            cleaned_block = "".join(kept).strip()
            if cleaned_block:
                cleaned_blocks.append(cleaned_block)

        return "\n\n".join(cleaned_blocks).strip()