from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from smolagents.monitoring import LogLevel

try:
    from ..citation_validator import CitationRecord
except Exception:
    from citation_validator import CitationRecord


class CitationFlowService:
    """引用治理服务：白名单约束、引用入库、验证流程。"""

    @staticmethod
    def enqueue_unverified_citation(agent, author: str, year: str, title: str, claim: str) -> None:
        """将引用加入待验证队列（按 author-year 去重）。"""
        author_norm = str(author or "").strip()
        year_norm = str(year or "").strip()
        if not author_norm or not year_norm:
            return

        queue_key = f"{author_norm.lower()}::{year_norm}"
        if queue_key in agent._queued_citation_keys:
            agent._log_reference_event(
                stage="step0_queue_dedup",
                section_title="待验证队列",
                detail=(
                    "步骤0/5（写作期-入队去重）\n"
                    f"队列键: {queue_key}\n"
                    "结果: 已存在，跳过"
                ),
            )
            return

        agent._queued_citation_keys.add(queue_key)
        agent._unverified_citations.append((author_norm, year_norm, str(title or "").strip(), str(claim or "").strip()))
        agent._log_reference_event(
            stage="step0_queue_enqueue",
            section_title="待验证队列",
            detail=(
                "步骤0/5（写作期-入待验证队列）\n"
                f"队列键: {queue_key}\n"
                f"作者: {author_norm}\n"
                f"年份: {year_norm}\n"
                f"题名: {str(title or '').strip() or 'N/A'}\n"
                f"claim: {str(claim or '').strip()[:300]}"
            ),
        )

    @staticmethod
    def extract_author_year_from_key(key: str) -> Tuple[str, str]:
        text = str(key or "").strip()
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return text, ""

    @staticmethod
    def get_first_author_surface(authors: str) -> str:
        raw = str(authors or "").strip()
        if not raw:
            return ""

        first = re.split(r"\s*(?:;|,|&| and )\s*", raw, maxsplit=1)[0].strip()
        if not first:
            first = raw

        parts = [p for p in re.split(r"\s+", first) if p]
        if not parts:
            return first

        if re.search(r"[A-Za-z]", first) and len(parts) > 1:
            return parts[-1].strip(".,")
        return first.strip(".,")

    @staticmethod
    def build_preferred_inline_citation(authors: str, year: str) -> str:
        authors_text = str(authors or "").strip()
        year_text = str(year or "").strip()
        if not authors_text or not year_text:
            return ""

        if "et al." in authors_text.lower():
            label = authors_text
        else:
            multi_author = bool(re.search(r"[,;&]", authors_text) or " and " in authors_text.lower())
            first_author = CitationFlowService.get_first_author_surface(authors_text)
            if multi_author and first_author:
                label = f"{first_author} et al."
            else:
                label = first_author or authors_text

        return f"({label}, {year_text})"

    @staticmethod
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

    @staticmethod
    def build_allowed_citation_alias_map(allowed_citation_keys: Optional[Set[str]]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        if not allowed_citation_keys:
            return alias_map

        for canonical_key in allowed_citation_keys:
            canonical = str(canonical_key or "").strip()
            if not canonical:
                continue

            authors, year = CitationFlowService.extract_author_year_from_key(canonical)
            if not authors or not year:
                continue

            variants = {canonical, f"({authors}, {year})"}
            first_author = CitationFlowService.get_first_author_surface(authors)
            if first_author:
                variants.add(f"{first_author} ({year})")
                variants.add(f"({first_author}, {year})")

            multi_author = bool(re.search(r"[,;&]", authors) or " and " in authors.lower())
            if multi_author and first_author:
                etal = f"{first_author} et al."
                variants.add(f"{etal} ({year})")
                variants.add(f"({etal}, {year})")

            for variant in variants:
                normalized = CitationFlowService.normalize_citation_surface(variant)
                if normalized:
                    alias_map[normalized] = canonical

        return alias_map

    @staticmethod
    def resolve_allowed_citation_key(raw_key: str, allowed_citation_keys: Optional[Set[str]]) -> Optional[str]:
        if not allowed_citation_keys:
            return None
        alias_map = CitationFlowService.build_allowed_citation_alias_map(allowed_citation_keys)
        normalized = CitationFlowService.normalize_citation_surface(raw_key)
        return alias_map.get(normalized)

    @staticmethod
    def scan_citations_in_text(text: str) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        seen_spans: Set[Tuple[int, int]] = set()
        content = str(text or "")

        parenthetical_pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
        narrative_pattern = r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-.]*(?:\s+(?:et al\.|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-.]*|AI|Research|Institute|Team|Face|Insights|Newsroom))*)\s*\(((?:19|20)\d{2}|n\.d\.)\)"

        for match in re.finditer(parenthetical_pattern, content):
            start, end = match.span()
            seen_spans.add((start, end))
            findings.append(
                {
                    "kind": "parenthetical",
                    "raw_key": f"{match.group(1).strip()} ({match.group(2).strip()})",
                    "matched_text": match.group(0),
                    "start": start,
                    "end": end,
                }
            )

        for match in re.finditer(narrative_pattern, content):
            start, end = match.span()
            if (start, end) in seen_spans:
                continue
            findings.append(
                {
                    "kind": "narrative",
                    "raw_key": f"{match.group(1).strip()} ({match.group(2).strip()})",
                    "matched_text": match.group(0),
                    "start": start,
                    "end": end,
                }
            )

        findings.sort(key=lambda item: item.get("start", 0))
        return findings

    @staticmethod
    def format_allowed_citation_whitelist(agent, available_citations: Dict[str, Dict[str, str]]) -> str:
        if not available_citations:
            return "（当前段落没有可用引用，禁止输出任何 APA 文内引用）"

        lines = []
        for idx, (key, info) in enumerate(available_citations.items(), 1):
            authors = str(info.get("authors") or CitationFlowService.extract_author_year_from_key(key)[0]).strip()
            year = str(info.get("year") or CitationFlowService.extract_author_year_from_key(key)[1]).strip()
            title = str(info.get("title", "")).strip() or "Untitled source"
            inline = str(info.get("inline_citation", "")).strip() or CitationFlowService.build_preferred_inline_citation(authors, year)
            lines.append(f"- [{idx}] 只允许使用: {inline} | canonical_key: {key} | title: {title}")
        return "\n".join(lines)

    @staticmethod
    def apply_citation_replacements(text: str, replacements: List[Tuple[int, int, str]]) -> str:
        updated = str(text or "")
        for start, end, value in sorted(replacements, key=lambda item: item[0], reverse=True):
            updated = updated[:start] + value + updated[end:]
        return updated

    @staticmethod
    def enforce_strict_citation_flow_on_section(
        agent,
        text: str,
        section_title: str,
        available_citations: Dict[str, Dict[str, str]],
    ) -> str:
        current = str(text or "")
        if not agent.strict_citation_flow:
            return current

        allowed_keys = set(available_citations.keys()) if available_citations else set()

        def analyze(content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            findings = CitationFlowService.scan_citations_in_text(content)
            invalid: List[Dict[str, Any]] = []
            for item in findings:
                canonical_key = CitationFlowService.resolve_allowed_citation_key(item["raw_key"], allowed_keys)
                item["canonical_key"] = canonical_key
                if allowed_keys:
                    if canonical_key is None:
                        invalid.append(item)
                elif findings:
                    invalid.append(item)
            return findings, invalid

        findings, invalid = analyze(current)
        if invalid:
            whitelist_text = CitationFlowService.format_allowed_citation_whitelist(agent, available_citations)
            invalid_preview = "\n".join(f"- {item['matched_text']}" for item in invalid[:12])
            agent._log_reference_event(
                stage="citation_strict_rewrite",
                section_title=section_title,
                detail=(
                    "检测到正文引用超出白名单，触发强约束修订\n"
                    f"非法/未解析引用数: {len(invalid)}\n"
                    f"白名单:\n{whitelist_text}\n"
                    f"待修订引用:\n{invalid_preview or '（无）'}"
                ),
            )

            rewrite_prompt = (
                f"【章节】\n{section_title}\n\n"
                f"【原文】\n{current}\n\n"
                f"【引用白名单】\n{whitelist_text}\n\n"
                "【强约束要求】\n"
                "1. 只允许使用白名单中的文内引用，且必须逐字使用白名单给出的括号形式。\n"
                "2. 不允许出现任何新的作者名、年份或未在白名单中的 APA 引用。\n"
                "3. 不允许叙述式引用（如 X et al. (2020)）；如需引用，只能保留括号式。\n"
                "4. 若某句缺乏白名单支持，请删除该引用并同步收缩或改写该句，禁止保留无依据的强断言。\n"
                "5. 保持原段落主题、语言风格和大致长度。\n\n"
                "请直接输出修订后的完整段落，不要解释。"
            )

            for _ in range(max(1, agent.strict_citation_revision_max_attempts)):
                try:
                    current = str(agent.execute_tool_call("section_revision", {"input": rewrite_prompt}))
                except Exception as e:
                    agent.logger.log(f"⚠️ 强引用流修订失败: {e}", level=LogLevel.ERROR)
                    break
                findings, invalid = analyze(current)
                if not invalid:
                    break

        findings, _ = analyze(current)
        replacements: List[Tuple[int, int, str]] = []
        for item in findings:
            canonical_key = item.get("canonical_key")
            if canonical_key:
                info = available_citations.get(canonical_key, {})
                authors = str(info.get("authors") or CitationFlowService.extract_author_year_from_key(canonical_key)[0]).strip()
                year = str(info.get("year") or CitationFlowService.extract_author_year_from_key(canonical_key)[1]).strip()
                preferred = str(info.get("inline_citation", "")).strip() or CitationFlowService.build_preferred_inline_citation(authors, year)
                if preferred and item["matched_text"] != preferred:
                    replacements.append((item["start"], item["end"], preferred))
            else:
                replacements.append((item["start"], item["end"], ""))

        if replacements:
            current = CitationFlowService.apply_citation_replacements(current, replacements)

        return current

    @staticmethod
    def extract_citations_from_rag(agent, rag_context: str) -> Dict[str, Dict[str, str]]:
        citations = {}
        if not rag_context:
            latest_payload = agent.state.get(agent.STATE_LATEST_TAGGED_SEARCH_PAYLOAD, {})
            if isinstance(latest_payload, dict):
                citations.update(agent._tagged_search_service.extract_citations_from_tagged_payload(agent, latest_payload))
            return citations

        latest_payload = agent.state.get(agent.STATE_LATEST_TAGGED_SEARCH_PAYLOAD, {})
        if isinstance(latest_payload, dict):
            citations.update(agent._tagged_search_service.extract_citations_from_tagged_payload(agent, latest_payload))

        if agent.strict_citation_flow:
            return citations

        pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"
        matches = re.findall(pattern, str(rag_context))

        placeholder_authors = {
            "张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"
        }

        for author_raw, year_raw in matches:
            author = author_raw.strip()
            year = year_raw.strip()
            author_norm = author.lower().replace(" ", "")
            if author in placeholder_authors or author_norm in placeholder_authors:
                continue
            if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
                continue

            key = f"{author} ({year})"
            if key not in citations:
                citations[key] = {
                    "authors": author,
                    "year": year,
                    "title": "Retrieved from search results",
                    "title_source": "rag_context",
                }

        return citations

    @staticmethod
    def add_citations(agent, citations: Dict[str, Dict[str, str]], section_title: str = "") -> None:
        for key, info in citations.items():
            match = re.match(r"^(.+?)\s*\(([^)]+)\)$", key)
            author = match.group(1).strip() if match else str(info.get("authors", "")).strip()
            year = match.group(2).strip() if match else str(info.get("year", "")).strip()

            agent._log_reference_event(
                stage="step0_extract_from_rag",
                section_title=section_title,
                detail=(
                    "步骤0/5（写作期-候选提取）\n"
                    f"来源: RAG\n"
                    f"引用键: {key}\n"
                    f"解析作者: {author or 'N/A'}\n"
                    f"解析年份: {year or 'N/A'}"
                ),
            )

            if key not in agent._citations:
                agent._citation_counter += 1
                agent._citations[key] = info
                agent._log_reference_addition(
                    source="rag_context",
                    section_title=section_title,
                    citation_key=key,
                    citation_info=info,
                )
            else:
                existing = agent._citations.get(key, {})
                existing_title = str(existing.get("title", "")).strip()
                incoming_title = str(info.get("title", "")).strip()
                improved = bool(incoming_title) and incoming_title.lower() not in {
                    "title unavailable",
                    "retrieved from search results",
                }
                should_update = improved and (not existing_title or existing_title.lower() in {
                    "title unavailable",
                    "retrieved from search results",
                })

                if should_update:
                    existing.update(info)
                    agent._citations[key] = existing
                    agent._log_reference_event(
                        stage="citation_enriched",
                        section_title=section_title,
                        detail=(
                            f"引用键: {key}\n"
                            f"旧标题: {existing_title or '（空）'}\n"
                            f"新标题: {incoming_title}"
                        ),
                    )
                else:
                    agent._log_reference_event(
                        stage="citation_reused",
                        section_title=section_title,
                        detail=f"引用已存在，跳过新增: {key}",
                    )

            if author and year:
                citation_title = str(agent._citations.get(key, {}).get("title") or info.get("title") or "").strip()
                claim = f"章节「{section_title or '未知章节'}」使用了该引用。"
                CitationFlowService.enqueue_unverified_citation(agent, author, year, citation_title, claim)

    @staticmethod
    def collect_citations_from_text(
        agent,
        text: str,
        section_title: str = "",
        allowed_citation_keys: Optional[Set[str]] = None,
        add_new_from_text: bool = False,
    ) -> None:
        if not text:
            return

        normalized_allowed_keys: Optional[Set[str]] = None
        if allowed_citation_keys is not None:
            normalized_allowed_keys = {
                str(k).strip() for k in allowed_citation_keys if str(k).strip()
            }

        pattern = r"\(([^()]{1,80}?),\s*((?:19|20)\d{2}|n\.d\.)\)"

        placeholder_authors = {
            "张三", "李四", "王五", "赵六", "某某", "佚名", "作者", "作者等", "author", "anonymous"
        }

        sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])\s+|\n+', str(text)) if s.strip()]
        for sentence in sentences:
            matches = re.findall(pattern, sentence)
            for author_raw, year_raw in matches:
                author = author_raw.strip()
                year = year_raw.strip()

                author_norm = author.lower().replace(" ", "")
                if author in placeholder_authors or author_norm in placeholder_authors:
                    continue
                if re.search(r"张三|李四|王五|赵六|某某|^作者$|^Author$", author, re.IGNORECASE):
                    continue

                key = f"{author} ({year})"
                resolved_key = CitationFlowService.resolve_allowed_citation_key(key, normalized_allowed_keys)
                agent._log_reference_event(
                    stage="step0_extract_from_text",
                    section_title=section_title,
                    detail=(
                        "步骤0/5（写作期-候选提取）\n"
                        "来源: generated_text\n"
                        f"引用键: {key}\n"
                        f"解析到白名单: {resolved_key or '否'}\n"
                        f"句子: {sentence}"
                    ),
                )

                if normalized_allowed_keys is not None and resolved_key is None:
                    agent._log_reference_event(
                        stage="citation_rejected_not_in_rag",
                        section_title=section_title,
                        detail=(
                            "正文出现了未在检索证据池中的引用，已拒绝入库\n"
                            f"引用键: {key}\n"
                            f"句子: {sentence}"
                        ),
                    )
                    continue

                if resolved_key:
                    key = resolved_key
                    author, year = CitationFlowService.extract_author_year_from_key(resolved_key)

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
                        agent._log_reference_addition(
                            source="generated_text",
                            section_title=section_title,
                            citation_key=key,
                            citation_info=agent._citations[key],
                        )
                    else:
                        agent._log_reference_event(
                            stage="citation_not_added_from_text",
                            section_title=section_title,
                            detail=(
                                "正文引用未入库（策略：禁止仅从正文新增）\n"
                                f"引用键: {key}\n"
                                f"句子: {sentence}"
                            ),
                        )
                        continue

                CitationFlowService.enqueue_unverified_citation(
                    agent,
                    author=author,
                    year=year,
                    title=str(agent._citations.get(key, {}).get("title") or "").strip(),
                    claim=sentence,
                )

    @staticmethod
    def validate_all_citations(agent) -> Dict[str, Any]:
        if not agent.enable_citation_validation:
            agent.logger.log(
                "⚠️ 引用验证功能未启用，跳过验证步骤\n"
                "若要启用，请调用 enable_citation_validation_mode(True)",
                level=LogLevel.INFO,
            )
            return {"status": "disabled", "message": "Citation validation not enabled"}

        if not agent._unverified_citations:
            agent.logger.log("ℹ️ 没有待验证的引用", level=LogLevel.INFO)
            return {"status": "empty", "citations_count": 0}

        agent.logger.log(
            f"🔍 开始验证 {len(agent._unverified_citations)} 个引用...",
            level=LogLevel.INFO,
        )
        agent._log_reference_event(
            stage="validation_start",
            section_title="验证阶段",
            detail=(
                "开始执行 5 步验证流程：\n"
                "Step 1 Search → Step 2 Verify → Step 3 Retrieve → Step 4 Validate → Step 5 Add\n"
                f"待验证条目数: {len(agent._unverified_citations)}"
            ),
        )

        citations_to_validate = {}
        for author, year, title, claim in agent._unverified_citations:
            key = f"{author}_{year}"
            if key not in citations_to_validate:
                citations_to_validate[key] = (author, year, title, claim)

        validation_results = {}
        for idx, (key, (author, year, title, claim)) in enumerate(citations_to_validate.items(), 1):
            agent.logger.log(
                f"[{idx}/{len(citations_to_validate)}] 验证: {author} ({year})",
                level=LogLevel.INFO,
            )
            agent._log_reference_event(
                stage="step1_search_start",
                section_title="验证阶段",
                detail=(
                    f"步骤1/5 Search 开始\n"
                    f"条目: {author} ({year})\n"
                    f"标题线索: {title or 'N/A'}"
                ),
            )

            search_results = agent._citation_validator.step1_search(author, year, title)
            agent._log_reference_event(
                stage="step1_search_done",
                section_title="验证阶段",
                detail=(
                    f"步骤1/5 Search 完成\n"
                    f"crossref: {len(search_results.get('crossref', []))}\n"
                    f"arxiv: {len(search_results.get('arxiv', []))}\n"
                    f"openalex: {len(search_results.get('openalex', []))}"
                ),
            )

            verified, verify_info = agent._citation_validator.step2_verify(search_results)
            agent._log_reference_event(
                stage="step2_verify_done",
                section_title="验证阶段",
                detail=(
                    f"步骤2/5 Verify 完成\n"
                    f"条目: {author} ({year})\n"
                    f"结果: {'通过' if verified else '失败'}\n"
                    f"sources_found: {verify_info.get('sources_found', [])}"
                ),
            )

            if not verified:
                agent.logger.log(
                    f"❌ {author} ({year}) 验证失败: 未找到双源",
                    level=LogLevel.INFO,
                )
                validation_results[key] = {
                    "status": "rejected",
                    "reason": "Failed dual-source verification",
                    "sources_found": verify_info.get("sources_found", []),
                }
                continue

            best_result = verify_info["best_result"]
            bibtex = agent._citation_validator.step3_retrieve(best_result)
            agent._log_reference_event(
                stage="step3_retrieve_done",
                section_title="验证阶段",
                detail=(
                    f"步骤3/5 Retrieve 完成\n"
                    f"条目: {author} ({year})\n"
                    f"best_title: {best_result.get('title', 'N/A')}\n"
                    f"bibtex获取: {'成功' if bool(bibtex) else '失败/为空'}"
                ),
            )

            record = CitationRecord(
                authors=author,
                year=year,
                title=title or best_result.get("title", "Unknown"),
                source=verify_info["sources_found"][0],
                url=best_result.get("url", ""),
                bibtex=bibtex or "",
                abstract=best_result.get("abstract", ""),
                verification_status="verified",
                claim_text=claim,
            )

            if record.abstract and claim:
                validation = agent._citation_validator.step4_validate(record, claim)
                record.validation_result = validation
                agent._log_reference_event(
                    stage="step4_validate_done",
                    section_title="验证阶段",
                    detail=(
                        f"步骤4/5 Validate 完成\n"
                        f"条目: {author} ({year})\n"
                        f"is_valid: {validation.get('is_valid')}\n"
                        f"confidence: {validation.get('confidence', 'N/A')}"
                    ),
                )

                if not validation.get("is_valid"):
                    record.verification_status = "rejected"
                    agent.logger.log(
                        f"⚠️ {author} ({year}) 摘要与声明不匹配",
                        level=LogLevel.INFO,
                    )
            else:
                agent._log_reference_event(
                    stage="step4_validate_skipped",
                    section_title="验证阶段",
                    detail=(
                        f"步骤4/5 Validate 跳过\n"
                        f"条目: {author} ({year})\n"
                        f"原因: {'缺少abstract' if not record.abstract else '缺少claim'}"
                    ),
                )

            success = agent._citation_validator.step5_add(record)
            agent._log_reference_event(
                stage="step5_add_done",
                section_title="验证阶段",
                detail=(
                    f"步骤5/5 Add 完成\n"
                    f"条目: {author} ({year})\n"
                    f"写入validator缓存: {'成功' if success else '失败'}\n"
                    f"最终状态: {record.verification_status}"
                ),
            )

            validation_results[key] = {
                "status": record.verification_status,
                "source": record.source,
                "bibtex": record.bibtex,
                "has_abstract": bool(record.abstract),
                "validation": record.validation_result,
            }

            if record.verification_status == "verified":
                CitationFlowService.add_validated_citation(agent, record)

        CitationFlowService.save_citation_validation_log(agent, validation_results)

        agent._unverified_citations.clear()
        agent._queued_citation_keys.clear()

        report = agent._citation_validator.get_validation_report()
        agent.logger.log(
            f"✅ 引用验证完成: {report['verified_count']}/{report['total_citations_processed']} 通过验证",
            level=LogLevel.INFO,
        )
        agent._log_reference_event(
            stage="validation_summary",
            section_title="验证阶段",
            detail=(
                "5 步验证结束\n"
                f"total: {len(citations_to_validate)}\n"
                f"verified: {report['verified_count']}\n"
                f"rejected: {report['rejected_count']}\n"
                f"verification_rate: {report['verification_rate']}"
            ),
        )

        return {
            "status": "completed",
            "total": len(citations_to_validate),
            "verified": report["verified_count"],
            "rejected": report["rejected_count"],
            "verification_rate": report["verification_rate"],
            "results": validation_results,
        }

    @staticmethod
    def add_validated_citation(agent, record: CitationRecord) -> None:
        key = f"{record.authors} ({record.year})"
        existing = agent._citations.get(key, {})
        if key not in agent._citations:
            agent._citation_counter += 1

        merged = dict(existing)
        merged.update(
            {
                "authors": record.authors,
                "year": record.year,
                "title": record.title,
                "source": record.source,
                "url": record.url,
                "bibtex": record.bibtex,
                "verified": True,
                "verification_sources": [record.source],
                "verification_status": "verified",
            }
        )
        if "id" not in merged:
            merged["id"] = agent._citation_counter
        agent._citations[key] = merged

    @staticmethod
    def save_citation_validation_log(agent, validation_results: Dict) -> None:
        try:
            log_data = {
                "timestamp": time.time(),
                "total_citations": len(validation_results),
                "results": validation_results,
                "validator_report": agent._citation_validator.get_validation_report(),
            }

            with open(agent._citations_validation_log, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            agent.logger.log(
                f"✅ 引用验证日志已保存: {agent._citations_validation_log}",
                level=LogLevel.INFO,
            )
        except Exception as e:
            agent.logger.log(f"⚠️ 保存引用验证日志失败: {e}", level=LogLevel.ERROR)


def main() -> int:
    parser = argparse.ArgumentParser(description="CitationFlowService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("CitationFlowService: use --help on component files for end-to-end debugging.")
        return 0

    print("这是 service 文件，不直接执行。")
    print("请改用组件文件调试，例如：")
    print("python examples/open_deep_research/scripts/long_writer/component_body.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
