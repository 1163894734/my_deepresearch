from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from smolagents.monitoring import LogLevel


@dataclass
class CitationRecord:
    authors: str = ""
    year: str = ""
    title: str = ""
    source: str = ""
    url: str = ""
    bibtex: str = ""
    abstract: str = ""
    verification_status: str = "pending"
    claim_text: str = ""
    validation_result: Dict[str, Any] = field(default_factory=dict)


class CitationFlowService:
    """引用治理服务：白名单约束、引用入库、验证流程。"""

    @staticmethod
    def build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
        """
        主要作用：构造引用条目的 canonical_key，用于内部索引、去重和验证。

        输入参数：
        - authors (str): 作者列表的字符串表示，用于构造文内引用、canonical_key 或参考文献条目。
        - year (str): 年份字符串，用于检索约束、canonical_key 构造和文内引用匹配。
        - fallback_title (str): 当缺少作者和年份时用于降级构造键值的标题。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前阶段最关键的上下文字段（任务、章节、材料、引用）。
        - 按工作流约定拼装提示词或结构化载荷。
        - 返回可直接交给下游组件执行的输入对象。
        """

        authors_text = str(authors or "").strip()
        year_text = str(year or "").strip()
        if authors_text and year_text:
            return f"{authors_text} ({year_text})"
        return str(fallback_title or "").strip()

    @staticmethod
    def enqueue_unverified_citation(agent, author: str, year: str, title: str, claim: str) -> None:
        """
        主要作用：将候选引用加入待验证队列并记录日志。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - author (str): 作者线索字符串，用于检索或引用入队。
        - year (str): 年份字符串，用于检索约束、canonical_key 构造和文内引用匹配。
        - title (str): 标题线索，可指论文标题、章节标题或搜索提示中的文献标题。
        - claim (str): 待验证或待记录的正文声明。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """
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
        """
        主要作用：从 canonical_key 中拆解作者和年份。

        输入参数：
        - key (str): 该参数用于承载 `key` 相关的业务上下文或控制信息。

        返回值：
        - Tuple[str, str]：返回多个并列结果，便于同时传递主结果与附加元数据。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
        """

        text = str(key or "").strip()
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return text, ""

    @staticmethod
    def normalize_available_citations(raw: Any) -> Dict[str, Dict[str, Any]]:
        """
        主要作用：把不同形态的 available_citations 统一为标题索引字典。

        输入参数：
        - raw (Any): 未经规范化的原始输入，可为字典、列表、字符串或混合结构。

        返回值：
        - Dict[str, Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
        """
        normalized: Dict[str, Dict[str, Any]] = {}

        if isinstance(raw, dict):
            for outer_key, value in raw.items():
                if isinstance(value, dict):
                    info = dict(value)
                else:
                    info = {"title": str(value or "").strip()}

                key_text = str(outer_key or "").strip()
                title = str(info.get("title") or "").strip()
                if not title:
                    if re.match(r"^.+?\s*\((?:19|20)\d{2}|n\.d\.\)$", key_text):
                        title = key_text
                    else:
                        title = key_text
                if not title:
                    continue

                info["title"] = title
                authors = str(info.get("authors", "")).strip()
                year = str(info.get("year", "")).strip()
                canonical_key = str(info.get("canonical_key", "")).strip() or CitationFlowService.build_canonical_citation_key(authors, year, title)
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
                canonical_key = CitationFlowService.build_canonical_citation_key(authors, year, title)
                if canonical_key:
                    info["canonical_key"] = canonical_key
                info["title"] = title
                normalized[title] = info
            return normalized

        return normalized

    @staticmethod
    def build_available_citation_maps(available_citations: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        主要作用：同时构造标题索引和 canonical_key 索引两套引用映射。

        输入参数：
        - available_citations (Any): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取当前阶段最关键的上下文字段（任务、章节、材料、引用）。
        - 按工作流约定拼装提示词或结构化载荷。
        - 返回可直接交给下游组件执行的输入对象。
        """
        title_map = CitationFlowService.normalize_available_citations(available_citations)
        canonical_map: Dict[str, Dict[str, Any]] = {}
        for title, info in title_map.items():
            if not isinstance(info, dict):
                continue
            row = dict(info)
            row.setdefault("title", title)
            authors = str(row.get("authors", "")).strip()
            year = str(row.get("year", "")).strip()
            canonical_key = str(row.get("canonical_key", "")).strip() or CitationFlowService.build_canonical_citation_key(authors, year, title)
            if not canonical_key:
                continue
            row["canonical_key"] = canonical_key
            canonical_map[canonical_key] = row
        return title_map, canonical_map

    @staticmethod
    def get_first_author_surface(authors: str) -> str:
        """
        主要作用：提取适合生成 APA 文内引用的第一作者表面形式。

        输入参数：
        - authors (str): 作者列表的字符串表示，用于构造文内引用、canonical_key 或参考文献条目。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """

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
        """
        主要作用：根据作者和年份生成首选括号式文内引用。

        输入参数：
        - authors (str): 作者列表的字符串表示，用于构造文内引用、canonical_key 或参考文献条目。
        - year (str): 年份字符串，用于检索约束、canonical_key 构造和文内引用匹配。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取当前阶段最关键的上下文字段（任务、章节、材料、引用）。
        - 按工作流约定拼装提示词或结构化载荷。
        - 返回可直接交给下游组件执行的输入对象。
        """

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
        """
        主要作用：将引用表面形式规范化，便于别名匹配。

        输入参数：
        - citation_text (str): 正文中的引用文本，可能是括号式或叙述式表面形式。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 扫描文本中的章节结构、引用模式或候选字段。
        - 按学术写作约束执行清洗、规范化与必要回填。
        - 生成可直接进入后续写作/验证/汇总流程的中间结果。
        """

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
        """
        主要作用：构建白名单引用的别名映射表。

        输入参数：
        - allowed_citation_keys (Optional[Set[str]]): 允许出现在正文中的 canonical_key 集合，用于白名单校验。

        返回值：
        - Dict[str, str]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取当前阶段最关键的上下文字段（任务、章节、材料、引用）。
        - 按工作流约定拼装提示词或结构化载荷。
        - 返回可直接交给下游组件执行的输入对象。
        """

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
        """
        主要作用：把正文中的引用表面形式解析成白名单 canonical_key。

        输入参数：
        - raw_key (str): 正文扫描得到的引用表面形式，用于解析为 canonical_key。
        - allowed_citation_keys (Optional[Set[str]]): 允许出现在正文中的 canonical_key 集合，用于白名单校验。

        返回值：
        - Optional[str]：返回该方法的主要输出结果。

        实现逻辑：
        - 读取待校验输入并应用当前业务约束。
        - 执行验证、判定、拒绝或映射逻辑。
        - 返回验证结果，或同步更新状态与日志。
        """

        if not allowed_citation_keys:
            return None
        alias_map = CitationFlowService.build_allowed_citation_alias_map(allowed_citation_keys)
        normalized = CitationFlowService.normalize_citation_surface(raw_key)
        return alias_map.get(normalized)

    @staticmethod
    def scan_citations_in_text(text: str) -> List[Dict[str, Any]]:
        """
        主要作用：扫描正文中的括号式和叙述式引用并返回位置明细。

        输入参数：
        - text (str): 待解析、清洗或重写的原始文本内容。

        返回值：
        - List[Dict[str, Any]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """

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
        """
        主要作用：将可用引用格式化为提示词和日志可读的白名单列表。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - available_citations (Dict[str, Dict[str, str]]): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """

        if not available_citations:
            return "（当前段落没有可用引用，禁止输出任何 APA 文内引用）"

        _, canonical_map = CitationFlowService.build_available_citation_maps(available_citations)
        lines = []
        for idx, (key, info) in enumerate(canonical_map.items(), 1):
            authors = str(info.get("authors") or CitationFlowService.extract_author_year_from_key(key)[0]).strip()
            year = str(info.get("year") or CitationFlowService.extract_author_year_from_key(key)[1]).strip()
            title = str(info.get("title", "")).strip() or "Untitled source"
            inline = str(info.get("inline_citation", "")).strip() or CitationFlowService.build_preferred_inline_citation(authors, year)
            lines.append(f"- [{idx}] 只允许使用: {inline} | canonical_key: {key} | title: {title}")
        return "\n".join(lines)

    @staticmethod
    def apply_citation_replacements(text: str, replacements: List[Tuple[int, int, str]]) -> str:
        """
        主要作用：根据位置批量替换正文中的引用文本。

        输入参数：
        - text (str): 待解析、清洗或重写的原始文本内容。
        - replacements (List[Tuple[int, int, str]]): 基于字符区间的替换计划列表，每项为 (start, end, replacement)。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 结合当前流程阶段读取关键输入并完成边界检查。
        - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
        - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
        """

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
        """
        主要作用：对章节执行严格引用治理，只保留白名单允许的文内引用。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - available_citations (Dict[str, Dict[str, str]]): 按文献标题索引的可用引用字典，值中包含 authors、year、title、canonical_key 和 inline_citation 等元数据。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 先扫描文本中的所有 APA 引用，并与白名单 canonical_key 做别名解析匹配。
        - 若存在越权引用，触发受限重写提示词，多轮修订直到非法引用清零或达到上限。
        - 最后统一替换为白名单首选括号格式；无法解析的引用直接删除。
        """

        current = str(text or "")
        if not agent.strict_citation_flow:
            return current

        _, canonical_map = CitationFlowService.build_available_citation_maps(available_citations)
        allowed_keys = set(canonical_map.keys()) if canonical_map else set()

        def analyze(content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            """
            主要作用：分析文本中的引用并区分有效与无效条目。

            输入参数：
            - content (str): 正文、章节或日志等具体文本主体。

            返回值：
            - Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]：返回结构化字典结果，便于后续工作流阶段继续消费。

            实现逻辑：
            - 结合当前流程阶段读取关键输入并完成边界检查。
            - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
            - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
            """

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
                info = canonical_map.get(canonical_key, {})
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
    def remove_rejected_citation_sentences(
        agent,
        text: str,
        section_title: str,
        rejected_keys: Set[str],
    ) -> Tuple[str, int]:
        """
        主要作用：删除包含已拒绝引用的整句。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - rejected_keys (Set[str]): 验证失败的 canonical_key 集合，用于句子级清理。

        返回值：
        - Tuple[str, int]：返回多个并列结果，便于同时传递主结果与附加元数据。

        实现逻辑：
        - 将正文拆分为句子，逐句扫描其中引用并映射到 rejected_keys。
        - 只要某句命中任一拒绝引用，就删除整句，避免保留无依据断言。
        - 输出清理后文本与删除计数，并记录逐句与汇总审计日志。
        """
        current = str(text or "")
        if not current or not rejected_keys:
            return current, 0

        removed_count = 0
        kept_sentences: List[str] = []
        sentences = [s for s in re.split(r'(?<=[。！？!?])\s+|\n+', current) if s and s.strip()]

        for sentence in sentences:
            findings = CitationFlowService.scan_citations_in_text(sentence)
            hit_keys: List[str] = []
            for item in findings:
                canonical_key = CitationFlowService.resolve_allowed_citation_key(item.get("raw_key", ""), rejected_keys)
                if canonical_key and canonical_key in rejected_keys:
                    hit_keys.append(canonical_key)

            if hit_keys:
                removed_count += 1
                agent._log_reference_event(
                    stage="citation_sentence_removed",
                    section_title=section_title,
                    detail=(
                        "因引用验证失败，删除包含该引用的整句\n"
                        f"rejected_keys: {sorted(set(hit_keys))}\n"
                        f"deleted_sentence: {str(sentence).strip()[:500]}"
                    ),
                )
            else:
                kept_sentences.append(sentence.strip())

        new_text = "\n".join([s for s in kept_sentences if s])
        if removed_count > 0:
            agent._log_reference_event(
                stage="citation_rejected_cleanup_summary",
                section_title=section_title,
                detail=(
                    "拒绝引用清理完成\n"
                    f"removed_sentence_count: {removed_count}\n"
                    f"remaining_length: {len(new_text)}"
                ),
            )

        return new_text, removed_count

    @staticmethod
    def extract_citations_from_rag(agent, rag_context: str) -> Dict[str, Dict[str, str]]:
        """
        主要作用：从 RAG 证据上下文中提取可用引用。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - rag_context (str): 检索增强生成阶段的证据上下文文本。

        返回值：
        - Dict[str, Dict[str, str]]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 根据输入条件构造检索查询或搜索策略。
        - 调用外部搜索源、网页代理或内部服务获取候选结果。
        - 对结果做清洗、去重和结构化整理后返回。
        """

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

    @staticmethod
    def add_citations(agent, citations: Dict[str, Dict[str, str]], section_title: str = "") -> None:
        """
        主要作用：将引用集合写入代理引用库，并同步加入待验证队列。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - citations (Dict[str, Dict[str, str]]): 待验证或待入库的引用集合。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 将输入规范化为 canonical_map，统一不同来源与键形态的引用表示。
        - 新引用入库、重复引用按“标题质量提升”规则选择性覆盖并记录事件。
        - 对可解析作者和年份的条目自动入待验证队列，衔接后续 5 步验证流程。
        """

        _, canonical_map = CitationFlowService.build_available_citation_maps(citations)
        for key, info in canonical_map.items():
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
        """
        主要作用：从生成文本中扫描引用，并按策略入队、入库或拒绝。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - text (str): 待解析、清洗或重写的原始文本内容。
        - section_title (str): 章节标题，用于检索、日志记录和输出文件定位。
        - allowed_citation_keys (Optional[Set[str]]): 允许出现在正文中的 canonical_key 集合，用于白名单校验。
        - add_new_from_text (bool): 该参数用于承载 `add_new_from_text` 相关的业务上下文或控制信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 逐句扫描括号式 APA 引用并过滤占位作者，生成候选 canonical_key。
        - 若配置白名单，仅允许白名单可解析引用继续流转，其余直接拒绝并打日志。
        - 对允许引用执行入库策略（可选）并入待验证队列，保存 claim 供 Step4 使用。
        """

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
        """
        主要作用：执行五步引用验证并汇总每条引用的最终状态。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 对待验证队列去重后逐条执行 Step1~Step5，并记录每一步可解释日志。
        - 在 Step2 或 Step4 失败时明确 rejection_step/reason，保证拒绝路径可追溯。
        - 汇总通过率与条目明细，落盘验证日志并回写代理状态用于后续章节治理。
        """

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
        item_summaries: List[str] = []
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

            first_author = str(author or "").split(";")[0].split(",")[0].strip() or str(author or "").strip()
            inline_citation = f"({first_author} et al., {year})" if first_author and year else ""
            if "contributors" in str(author or "").lower() and year:
                inline_citation = f"({author}, {year})"

            title_key = str(title or f"{author} ({year})").strip() or f"{author}_{year}"
            claim_text = str(claim or "").strip()
            paragraph_for_validation = f"{claim_text} {inline_citation}".strip()
            citation_payload = {
                title_key: {
                    "title": str(title or "").strip() or title_key,
                    "authors": str(author or "").strip(),
                    "year": str(year or "").strip(),
                    "apa_citation": inline_citation,
                    "source": ["crossref", "arxiv"],
                }
            }

            search_results, _paragraph_after_step1 = agent._citation_validator.step1_search(citation_payload, paragraph_for_validation)
            agent._log_reference_event(
                stage="step1_search_done",
                section_title="验证阶段",
                detail=(
                    f"步骤1/5 Search 完成\n"
                    f"条目数: {len(search_results)}\n"
                    f"citation_key: {title_key}"
                ),
            )

            step2_map, _paragraph_after_step2 = agent._citation_validator.step2_verify(search_results, paragraph_for_validation)
            step2_row = step2_map.get(title_key, {}) if isinstance(step2_map, dict) else {}
            sources_found = step2_row.get("source", []) if isinstance(step2_row.get("source", []), list) else []
            source_hit_count = len(sources_found)
            matched_source_count = source_hit_count
            verification_counter = str(step2_row.get("verification_counter") or f"{source_hit_count}/{getattr(agent._citation_validator, 'min_source_count', 2)}")
            verification_reason = "enough_sources" if str(step2_row.get("verification_status", "")).lower() == "verified" else "insufficient_sources"
            verified = str(step2_row.get("verification_status", "")).lower() == "verified"
            agent._log_reference_event(
                stage="step2_verify_done",
                section_title="验证阶段",
                detail=(
                    f"步骤2/5 Verify 完成\n"
                    f"条目: {author} ({year})\n"
                    f"结果: {'通过' if verified else '失败'}\n"
                    f"verification_counter: {verification_counter}\n"
                    f"verification_reason: {verification_reason}\n"
                    f"matched_source_count: {matched_source_count}\n"
                    f"source_hit_count: {source_hit_count}\n"
                    f"sources_found: {sources_found}"
                ),
            )

            if not verified:
                agent.logger.log(
                    f"❌ {author} ({year}) 验证失败: 有效来源不足",
                    level=LogLevel.INFO,
                )
                validation_results[key] = {
                    "status": "rejected",
                    "reason": "Failed source verification",
                    "sources_found": sources_found,
                    "verification_counter": verification_counter,
                    "verification_reason": verification_reason,
                    "source_hit_count": source_hit_count,
                    "matched_source_count": matched_source_count,
                }
                item_summaries.append(
                    f"- {author} ({year}) => rejected | reason: source verification failed | counter: {verification_counter} | sources_found: {sources_found}"
                )
                agent._log_reference_event(
                    stage="validation_item_result",
                    section_title="验证阶段",
                    detail=(
                        "单条验证结果\n"
                        f"条目: {author} ({year})\n"
                        "status: rejected\n"
                        "reason: source verification failed\n"
                        f"verification_counter: {verification_counter}\n"
                        f"verification_reason: {verification_reason}\n"
                        f"matched_source_count: {matched_source_count}\n"
                        f"source_hit_count: {source_hit_count}\n"
                        f"sources_found: {sources_found}"
                    ),
                )
                continue

            step3_map, _paragraph_after_step3 = agent._citation_validator.step3_retrieve(step2_map, paragraph_for_validation)
            step3_row = step3_map.get(title_key, {}) if isinstance(step3_map, dict) else {}
            bibtex = str(step3_row.get("bibtex") or "")
            agent._log_reference_event(
                stage="step3_retrieve_done",
                section_title="验证阶段",
                detail=(
                    f"步骤3/5 Retrieve 完成\n"
                    f"条目: {author} ({year})\n"
                    f"source: {step3_row.get('source', [])}\n"
                    f"title: {step3_row.get('title', 'N/A')}\n"
                    f"bibtex获取: {'成功' if bool(bibtex) else '失败/为空'}"
                ),
            )

            record = CitationRecord(
                authors=author,
                year=year,
                title=title or str(step3_row.get("title") or "Unknown"),
                source=", ".join(sources_found),
                url=str(step3_row.get("url") or ""),
                bibtex=bibtex or "",
                abstract=str(step3_row.get("abstract") or ""),
                verification_status="verified",
                claim_text=claim,
            )

            rejection_step: Optional[str] = None
            rejection_reason: Optional[str] = None

            step4_map, paragraph_after_step4 = agent._citation_validator.step4_validate(step3_map, paragraph_for_validation)
            step4_row = step4_map.get(title_key, {}) if isinstance(step4_map, dict) else {}
            validation = step4_row.get("validation_result", {}) if isinstance(step4_row, dict) else {}
            record.validation_result = validation if isinstance(validation, dict) else {}
            is_valid = bool(step4_row.get("claim_supported", False))
            confidence = float(record.validation_result.get("confidence", 0.0) or 0.0)
            val_reason = str(record.validation_result.get("reason") or "")
            claim_preview = str(claim or "")[:300]

            step4_outcome = f"✅ 行内引用匹配通过 (confidence={confidence})" if is_valid else f"❌ 行内引用匹配失败 (confidence={confidence})"
            step4_action = "通过 → 继续 Step 5" if is_valid else "拒绝该引用，不加入参考文献列表"

            agent._log_reference_event(
                stage="step4_validate_done",
                section_title="验证阶段",
                detail=(
                    f"步骤4/5 Validate 完成\n"
                    f"条目: {author} ({year})\n"
                    f"结果: {step4_outcome}\n"
                    f"action: {step4_action}\n"
                    f"reason: {val_reason or 'N/A'}\n"
                    f"claim: {claim_preview}"
                ),
            )

            if not is_valid:
                record.verification_status = "rejected"
                rejection_step = "step4_validate"
                rejection_reason = val_reason or "apa-inline-citation-match-failed"

            step5_map, _paragraph_after_step5 = agent._citation_validator.step5_add(step4_map, paragraph_after_step4)
            success = title_key in step5_map
            if success and record.verification_status == "verified":
                step5_action = "✅ 已加入已验证引用库（将出现在参考文献列表）"
            else:
                step5_action = "❌ 拒绝：不加入参考文献列表（正文引用标记原样保留，不生成参考文献条目）"
                record.verification_status = "rejected"
                if not rejection_step:
                    rejection_step = "step5_add"
                    rejection_reason = "verification_status 非 verified"

            agent._log_reference_event(
                stage="step5_add_done",
                section_title="验证阶段",
                detail=(
                    f"步骤5/5 Add 完成\n"
                    f"条目: {author} ({year})\n"
                    f"写入validator缓存: {'成功' if success else '失败'}\n"
                    f"最终状态: {record.verification_status}\n"
                    f"处理结果: {step5_action}"
                ),
            )

            validation_results[key] = {
                "status": record.verification_status,
                "source": record.source,
                "bibtex": record.bibtex,
                "has_abstract": bool(record.abstract),
                "validation": record.validation_result,
                "rejection_step": rejection_step,
                "rejection_reason": rejection_reason,
            }

            item_summaries.append(
                f"- {author} ({year}) => {record.verification_status} | source: {record.source or 'N/A'}"
                + (f" | rejected_at: {rejection_step}" if rejection_step else "")
            )

            # 汇总日志：明确说明最终状态及原因
            result_detail_lines = [
                "单条验证结果",
                f"条目: {author} ({year})",
                f"status: {record.verification_status}",
                f"source: {record.source or 'N/A'}",
                f"has_abstract: {bool(record.abstract)}",
                f"处理结果: {step5_action}",
            ]
            if rejection_step:
                result_detail_lines.append(f"拒绝发生在: {rejection_step}")
            if rejection_reason:
                result_detail_lines.append(f"拒绝原因: {rejection_reason}")
            if record.verification_status == "rejected" and record.claim_text:
                result_detail_lines.append(f"claim: {str(record.claim_text or '')[:200]}")
            if record.verification_status == "rejected" and record.abstract:
                result_detail_lines.append(f"abstract: {str(record.abstract or '')[:200]}")

            agent._log_reference_event(
                stage="validation_item_result",
                section_title="验证阶段",
                detail="\n".join(result_detail_lines),
            )

            if record.verification_status == "verified":
                CitationFlowService.add_validated_citation(agent, record)

        CitationFlowService.save_citation_validation_log(agent, validation_results)

        agent._unverified_citations.clear()
        agent._queued_citation_keys.clear()

        verified_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "verified")
        rejected_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "rejected")
        total_count = len(citations_to_validate)
        verification_rate = round((verified_count / total_count) * 100, 2) if total_count else 0.0
        report = {
            "total": total_count,
            "verified": verified_count,
            "rejected": rejected_count,
            "results": validation_results,
        }
        agent.logger.log(
            f"✅ 引用验证完成: {verified_count}/{total_count} 通过验证",
            level=LogLevel.INFO,
        )
        agent._log_reference_event(
            stage="validation_summary",
            section_title="验证阶段",
            detail=(
                "5 步验证结束\n"
                f"total: {total_count}\n"
                f"verified: {verified_count}\n"
                f"rejected: {rejected_count}\n"
                f"verification_rate: {verification_rate}%\n"
                "items:\n"
                f"{chr(10).join(item_summaries) if item_summaries else '- (无)'}"
            ),
        )

        return {
            "status": "completed",
            "total": total_count,
            "verified": verified_count,
            "rejected": rejected_count,
            "verification_rate": verification_rate,
            "results": validation_results,
            "validator_report": report,
        }

    @staticmethod
    def add_validated_citation(agent, record: CitationRecord) -> None:
        """
        主要作用：把单条验证通过的引用写入引用库。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - record (CitationRecord): 已验证的引用记录对象，包含标题、作者、摘要和验证状态等信息。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 读取待校验输入并应用当前业务约束。
        - 执行验证、判定、拒绝或映射逻辑。
        - 返回验证结果，或同步更新状态与日志。
        """

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
        """
        主要作用：保存引用验证结果与文本变更日志。

        输入参数：
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - validation_results (Dict): 批量引用验证结果字典，用于汇总和落盘。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        try:
            verified_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "verified")
            rejected_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "rejected")
            current_run = {
                "timestamp": time.time(),
                "total_citations": len(validation_results),
                "results": validation_results,
                "validator_report": {
                    "total": len(validation_results),
                    "verified": verified_count,
                    "rejected": rejected_count,
                    "results": validation_results,
                },
            }

            existing_data: Dict[str, Any] = {}
            try:
                with open(agent._citations_validation_log, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_data = loaded
            except Exception:
                existing_data = {}

            validation_runs = existing_data.get("validation_runs", [])
            if not isinstance(validation_runs, list):
                validation_runs = []
            validation_runs.append(current_run)

            section_text_logs = existing_data.get("section_text_logs", [])
            if not isinstance(section_text_logs, list):
                section_text_logs = []

            log_data = {
                "timestamp": current_run["timestamp"],
                "total_citations": current_run["total_citations"],
                "results": current_run["results"],
                "validator_report": current_run["validator_report"],
                "validation_runs": validation_runs,
                "section_text_logs": section_text_logs,
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
    """
    主要作用：执行 main 相关逻辑。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 结合当前流程阶段读取关键输入并完成边界检查。
    - 执行与长文写作、检索编排或引用治理直接相关的核心步骤。
    - 产出可被下游阶段消费的结果，并同步更新状态、日志与落盘文件。
    """

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
