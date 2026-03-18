from __future__ import annotations

import sys
from typing import Any, Dict

try:
    from .base_component import JsonWorkflowComponent, run_component_cli
except ImportError:
    from base_component import JsonWorkflowComponent, run_component_cli


class CitationValidationComponent(JsonWorkflowComponent):
    """引用验证组件。

    输入 JSON 示例:
    {}

    输出 JSON 示例:
    {
        "validation_result": {
            "status": "completed",
            "total": 8,
            "verified": 6,
            "rejected": 2
        }
    }
    """

    name = "citation_validation"
    description = "引用验证组件：执行 Search→Verify→Retrieve→Validate→Add 的验证流程。"
    input_format = {}
    output_format = {
        "validation_result": "Dict",
    }

    def run(self, agent, payload: dict) -> dict:
        """
        主要作用：执行当前组件的主流程，处理输入并返回结构化输出。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - dict：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        if not agent.enable_citation_validation:
            return {
                "validation_result": {
                    "status": "disabled",
                    "message": "Citation validation not enabled",
                }
            }

        if not agent._unverified_citations:
            return {"validation_result": {"status": "empty", "citations_count": 0}}

        citations_to_validate: Dict[str, Any] = {}
        for author, year, title, claim in agent._unverified_citations:
            key = f"{author}_{year}"
            if key not in citations_to_validate:
                citations_to_validate[key] = (author, year, title, claim)

        validation_results: Dict[str, Any] = {}
        for key, (author, year, title, claim) in citations_to_validate.items():
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

            step1_map, _ = agent._citation_validator.step1_search(citation_payload, paragraph_for_validation)
            step2_map, _ = agent._citation_validator.step2_verify(step1_map, paragraph_for_validation)
            step2_row = step2_map.get(title_key, {}) if isinstance(step2_map, dict) else {}
            verified = str(step2_row.get("verification_status", "")).lower() == "verified"
            sources_found = step2_row.get("source", []) if isinstance(step2_row.get("source", []), list) else []
            verification_counter = str(
                step2_row.get("verification_counter")
                or f"{len(sources_found)}/{getattr(agent._citation_validator, 'min_source_count', 2)}"
            )

            if not verified:
                validation_results[key] = {
                    "status": "rejected",
                    "reason": "Failed source verification",
                    "sources_found": sources_found,
                    "verification_counter": verification_counter,
                    "rejection_step": "step2_verify",
                    "rejection_reason": "insufficient_sources",
                }
                continue

            step3_map, _ = agent._citation_validator.step3_retrieve(step2_map, paragraph_for_validation)
            step4_map, paragraph_after_step4 = agent._citation_validator.step4_validate(step3_map, paragraph_for_validation)
            step4_row = step4_map.get(title_key, {}) if isinstance(step4_map, dict) else {}
            validation = step4_row.get("validation_result", {}) if isinstance(step4_row, dict) else {}
            is_valid = bool(step4_row.get("claim_supported", False))

            if not is_valid:
                validation_results[key] = {
                    "status": "rejected",
                    "source": ", ".join(sources_found),
                    "has_abstract": bool(step4_row.get("abstract")),
                    "validation": validation if isinstance(validation, dict) else {},
                    "rejection_step": "step4_validate",
                    "rejection_reason": str((validation if isinstance(validation, dict) else {}).get("reason") or "apa-inline-citation-match-failed"),
                }
                continue

            step5_map, _ = agent._citation_validator.step5_add(step4_map, paragraph_after_step4)
            success = title_key in step5_map
            step3_row = step3_map.get(title_key, {}) if isinstance(step3_map, dict) else {}
            if success:
                canonical_key = f"{author} ({year})"
                existing = agent._citations.get(canonical_key, {})
                if canonical_key not in agent._citations:
                    agent._citation_counter += 1
                merged = dict(existing)
                merged.update(
                    {
                        "authors": author,
                        "year": year,
                        "title": title or str(step3_row.get("title") or "Unknown"),
                        "source": ", ".join(sources_found),
                        "url": str(step3_row.get("url") or ""),
                        "bibtex": str(step3_row.get("bibtex") or ""),
                        "verified": True,
                        "verification_sources": [", ".join(sources_found)],
                        "verification_status": "verified",
                    }
                )
                if "id" not in merged:
                    merged["id"] = agent._citation_counter
                agent._citations[canonical_key] = merged

                validation_results[key] = {
                    "status": "verified",
                    "source": ", ".join(sources_found),
                    "bibtex": str(step3_row.get("bibtex") or ""),
                    "has_abstract": bool(step3_row.get("abstract")),
                    "validation": validation if isinstance(validation, dict) else {},
                    "rejection_step": None,
                    "rejection_reason": None,
                }
            else:
                validation_results[key] = {
                    "status": "rejected",
                    "source": ", ".join(sources_found),
                    "bibtex": str(step3_row.get("bibtex") or ""),
                    "has_abstract": bool(step3_row.get("abstract")),
                    "validation": validation if isinstance(validation, dict) else {},
                    "rejection_step": "step5_add",
                    "rejection_reason": "verification_status non-verified",
                }

        agent._unverified_citations.clear()
        agent._queued_citation_keys.clear()

        verified_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "verified")
        rejected_count = sum(1 for row in validation_results.values() if str(row.get("status", "")).lower() == "rejected")
        total_count = len(citations_to_validate)
        verification_rate = round((verified_count / total_count) * 100, 2) if total_count else 0.0

        return {
            "validation_result": {
                "status": "completed",
                "total": total_count,
                "verified": verified_count,
                "rejected": rejected_count,
                "verification_rate": verification_rate,
                "results": validation_results,
                "validator_report": {
                    "total": total_count,
                    "verified": verified_count,
                    "rejected": rejected_count,
                    "results": validation_results,
                },
            }
        }


def main() -> int:
    """
    主要作用：执行 main 相关逻辑。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """

    if len(sys.argv) <= 1:
        payload_json_text = r'''{}'''
        return run_component_cli(
            CitationValidationComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(CitationValidationComponent())


if __name__ == "__main__":
    raise SystemExit(main())
