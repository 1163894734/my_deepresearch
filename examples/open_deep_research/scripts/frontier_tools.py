import json
import os
import re
import ast
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
from smolagents import Tool
from duckduckgo_search import DDGS

import utils.common_utils as common_utils

import requests
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures

class MultiSourceDataCollectorTool(Tool):
    name = "tool_multi_source_data_collector"
    description = (
        "多源真实数据采集工具。通过 OpenAlex/arXiv 采集论文，PatentsView(USPTO) 采集专利，"
        "DuckDuckGo 采集科技新闻。输出符合规范的分组字典。"
    )
    inputs = {
        "domain": {
            "type": "string",
            "description": "目标采集的技术领域或主题关键词，例如 'Brain-Computer Interface'。",
        },
        "max_items_per_source": {
            "type": "integer",
            "description": "每个渠道最大采集数量，默认 10。",
            "nullable": True,
        },
        "paper_engine": {
            "type": "string",
            "description": "论文检索引擎，可选 'openalex' 或 'arxiv'。默认 'openalex'。",
            "nullable": True,
        },
        "patent_engine": {
            "type": "string",
            "description": "专利检索引擎，可选 'uspto' 或 'epo'。默认 'uspto'。",
            "nullable": True,
        }
    }
    output_type = "any"

    def forward(
        self, 
        domain: str, 
        max_items_per_source: int = 10,
        paper_engine: str = "openalex",
        patent_engine: str = "uspto"
    ) -> dict:
        limit = int(max_items_per_source) if max_items_per_source else 10
        paper_engine = (paper_engine or "openalex").lower()
        patent_engine = (patent_engine or "uspto").lower()

        print(f"[Data Collector] 开始跨源采集: 主题='{domain}', 论文源={paper_engine}, 专利源={patent_engine}")

        # 1. 采集学术论文
        papers = self._collect_papers(domain, limit, paper_engine)
        
        # 2. 采集专利
        patents = self._collect_patents(domain, limit, patent_engine)
        
        # 3. 采集科技新闻/网页 (使用 DuckDuckGo)
        web_data = self._collect_web_news(domain, limit)
        
        # 4. 行业报告与项目数据 (依用户要求暂时留空)
        reports = []
        projects = []

        aggregated_data = {
            "paper": papers,
            "patent": patents,
            "web": web_data,
            "report": reports,
            "project": projects
        }

        total = sum(len(v) for v in aggregated_data.values())
        print(f"[Data Collector] 采集完毕，共获取 {total} 条有效记录。")

        return aggregated_data

    def _collect_papers(self, query: str, limit: int, engine: str) -> list[dict]:
        papers = []
        try:
            if engine == "arxiv":
                encoded_query = urllib.parse.quote(f"all:{query}")
                url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={limit}"
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                root = ET.fromstring(response.text)
                ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('arxiv:entry', ns):
                    title = entry.find('arxiv:title', ns).text.replace('\n', ' ')
                    abstract = entry.find('arxiv:summary', ns).text.replace('\n', ' ')
                    published = entry.find('arxiv:published', ns).text[:10]
                    authors = [a.find('arxiv:name', ns).text for a in entry.findall('arxiv:author', ns)]
                    paper_id = entry.find('arxiv:id', ns).text
                    
                    papers.append({
                        "id": paper_id,
                        "title": title.strip(),
                        "author": ", ".join(authors),
                        "publish_time": published,
                        "source": "arXiv",
                        "abstract": abstract.strip(),
                        "url": paper_id
                    })
            else:
                url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per-page={limit}"
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                for work in data.get("results", []):
                    authors = [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])]
                    institutions = []
                    for a in work.get("authorships", []):
                        for inst in a.get("institutions", []):
                            if inst.get("display_name") and inst.get("display_name") not in institutions:
                                institutions.append(inst.get("display_name"))
                                
                    source = work.get("primary_location", {}).get("source", {})
                    source_name = source.get("display_name") if source else "OpenAlex Work"

                    abstract = ""
                    inv_abs = work.get("abstract_inverted_index")
                    if inv_abs:
                        word_index = []
                        for word, positions in inv_abs.items():
                            for pos in positions:
                                word_index.append((pos, word))
                        word_index.sort(key=lambda x: x[0])
                        abstract = " ".join([w[1] for w in word_index])

                    papers.append({
                        "id": work.get("id"),
                        "title": work.get("title", ""),
                        "author": ", ".join(filter(None, authors)),
                        "affiliation": " | ".join(institutions),
                        "publish_time": work.get("publication_date", ""),
                        "source": source_name,
                        "abstract": abstract,
                        "citation_count": work.get("cited_by_count", 0),
                        "url": work.get("doi") or work.get("id")
                    })
        except Exception as e:
            print(f"[Warning] 论文采集失败 ({engine}): {str(e)}")
            
        return papers

    def _collect_patents(self, query: str, limit: int, engine: str) -> list[dict]:
        patents = []
        try:
            if engine == "epo":
                # 保留正常降级：缺少鉴权回退到 USPTO
                print("[Notice] 欧洲专利局(EPO)接口需要配置 Consumer Key 与 Secret。系统已回退使用 USPTO 检索。")
                engine = "uspto"

            if engine == "uspto":
                q_json = json.dumps({"_text_any": {"patent_title": query}})
                f_json = json.dumps(["patent_number", "patent_title", "patent_date", "patent_abstract", "assignee_organization"])
                url = f"https://api.patentsview.org/patents/query?q={urllib.parse.quote(q_json)}&f={urllib.parse.quote(f_json)}&o={json.dumps({'per_page': limit})}"
                
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                for p in data.get("patents", []):
                    assignees = [a.get("assignee_organization", "") for a in p.get("assignees", []) if a.get("assignee_organization")]
                    
                    patents.append({
                        "id": p.get("patent_number"),
                        "title": p.get("patent_title", ""),
                        "affiliation": ", ".join(assignees),
                        "publish_time": p.get("patent_date", ""),
                        "source": "USPTO PatentsView",
                        "abstract": p.get("patent_abstract", ""),
                        "url": f"https://patents.google.com/patent/US{p.get('patent_number')}"
                    })
        except Exception as e:
            print(f"[Warning] 专利采集失败 ({engine}): {str(e)}")
            
        return patents

    def _collect_web_news(self, query: str, limit: int) -> list[dict]:
        web_data = []
        try:
            with DDGS() as ddgs:
                # 保留正常降级：尝试新闻搜索，没有就用网页搜索
                results = list(ddgs.news(query, max_results=limit))
                
                if not results:
                    results = list(ddgs.text(query, max_results=limit))

                for i, r in enumerate(results):
                    web_data.append({
                        "id": f"web_ddgs_{i}",
                        "title": r.get("title", ""),
                        "source": r.get("source", "Web Search"),
                        "publish_time": r.get("date", r.get("published", "")),
                        "abstract": r.get("body", ""),
                        "url": r.get("href", ""),
                        "key_viewpoints": r.get("body", "")
                    })
        except ImportError:
            print("[Error] 缺失 duckduckgo_search 库。请执行: pip install duckduckgo-search")
        except Exception as e:
            print(f"[Warning] 网页新闻采集失败: {str(e)}")
            
        return web_data


class FragmentedInfoAggregatorTool(Tool):
    name = "tool_fragmented_info_aggregator"
    description = (
        "碎片化信息聚合工具。将论文、专利、项目、报告、网页等多源碎片化信息进行归一化处理，"
        "输出统一的结构化技术数据表(JSON/Excel/Markdown)。"
    )
    inputs = {
        "multi_source_data": {
            "type": "any",
            "description": "多源数据。支持 list[dict] 或按来源分组的 dict。",
        },
        "enable_llm_enrichment": {
            "type": "boolean",
            "description": "是否开启大模型智能填充？开启后将自动阅读摘要，补全缺失的'核心方法'、'创新点'等字段。默认 True。",
            "nullable": True,
        },
        "output_formats": {
            "type": "array",
            "description": "输出格式列表，可选 json/excel/markdown。默认全部输出。",
            "nullable": True,
        },
        "out_dir": {
            "type": "string",
            "description": "输出目录。为空时不落盘，仅返回结构化结果。",
            "nullable": True,
        },
        "file_stem": {
            "type": "string",
            "description": "输出文件名前缀，默认 fragmented_tech_table。",
            "nullable": True,
        },
    }
    output_type = "any"

    CORE_FIELDS = [
        "标题",
        "来源",
        "作者与单位",
        "发表时间",
        "所属技术领域",
        "技术主题",
        "关键问题/目标",
        "核心方法/架构/方案",
        "关键效果/指标/成果",
        "创新点",
        "局限性/风险/不足",
        "潜在应用场景/落地价值",
        "未来演进方向与研究热点",
    ]

    EXT_FIELDS = {
        "paper": [
            "摘要（中英文）",
            "数据集与实验环境",
            "与SOTA对比结果",
            "引用量/影响力",
            "实验结论",
        ],
        "patent": [
            "技术分类号（IPC/CPC/领域编码）",
            "权利要求核心点（创新点拆解）",
            "专利价值指标（引用、家族数量、法律状态）",
        ],
        "project": [
            "项目类别（国家/省部/企业/横向）",
            "研究目标与考核指标",
            "技术路线图",
            "成果产出体系（论文、专利、样机、产品）",
            "经费规模/投入强度",
            "项目执行阶段状态",
        ],
        "report": [
            "行业规模与增长率",
            "政策驱动因素",
            "关键玩家与生态格局",
            "技术需求、痛点、瓶颈",
            "应用落地节奏",
            "风险评估（技术、市场、政策）",
        ],
        "web": [
            "关键观点/判断",
            "事件类型（发布、会议、突破、争议）",
            "关联技术/公司/人物",
            "动态情感倾向（正面/负面/中性）",
        ],
    }

    SOURCE_ALIASES = {
        "paper": {"paper", "论文", "article", "journal", "conference"},
        "patent": {"patent", "专利"},
        "project": {"project", "项目", "课题"},
        "report": {"report", "行业报告", "白皮书", "报告"},
        "web": {"web", "news", "blog", "网页", "动态", "webnews"},
    }

    SENTIMENT_ALIASES = {
        "positive": "正面",
        "negative": "负面",
        "neutral": "中性",
        "pos": "正面",
        "neg": "负面",
    }

    def forward(
        self,
        multi_source_data,
        enable_llm_enrichment: bool = True,
        output_formats: list | None = None,
        out_dir: str | None = None,
        file_stem: str | None = None,
    ) -> dict:
        records = self._flatten_records(multi_source_data)
        normalized_rows = []
        
        # 1. 基础映射
        for index, record in enumerate(records, start=1):
            row = self._normalize_record(record)
            row["记录ID"] = row.get("记录ID") or f"record_{index:04d}"
            normalized_rows.append(row)

        # 2. LLM 智能补全核心空置字段 (并发改造版)
        if enable_llm_enrichment and normalized_rows:
            print(f"\n[Aggregator] 🚀 开启大模型并发智能补全，准备深度阅读 {len(normalized_rows)} 条记录以填充空缺字段...")
            
            # 包装函数以便在多线程中保留原有的索引顺序
            def process_row_wrapper(item):
                idx, row = item
                try:
                    # 调用大模型提取逻辑
                    enriched_row = self._llm_enrich_empty_fields(row, idx + 1, len(normalized_rows))
                    return idx, enriched_row
                except Exception as e:
                    print(f"[Warning] 记录 {idx+1} 补全失败: {str(e)}")
                    return idx, row

            # 使用 ThreadPoolExecutor 控制并发数（建议设置为 5-10，视 API 并发限流限制而定）
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                items = list(enumerate(normalized_rows))
                results = list(executor.map(process_row_wrapper, items))
            
            # 按原始顺序重新赋值
            results.sort(key=lambda x: x[0])
            normalized_rows = [res[1] for res in results]

        # 3. 格式化输出
        table_columns = ["记录ID", "数据源类型"] + self.CORE_FIELDS + self._all_extension_fields()
        df = pd.DataFrame(normalized_rows)
        if df.empty:
            df = pd.DataFrame(columns=table_columns)
        else:
            for col in table_columns:
                if col not in df.columns:
                    df[col] = ""
            df = df[table_columns]
            df = df.fillna("")

        formats = self._normalize_output_formats(output_formats)
        stem = file_stem or "fragmented_tech_table"
        file_paths = {}

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            if "json" in formats:
                json_path = os.path.join(out_dir, f"{stem}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
                file_paths["json"] = json_path
            if "excel" in formats:
                excel_path = os.path.join(out_dir, f"{stem}.xlsx")
                df.to_excel(excel_path, index=False)
                file_paths["excel"] = excel_path
            if "markdown" in formats:
                md_path = os.path.join(out_dir, f"{stem}.md")
                markdown_text = self._to_markdown(df)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)
                file_paths["markdown"] = md_path

        return {
            "summary": {
                "total_records": int(len(df)),
                "llm_enriched": enable_llm_enrichment,
                "by_source": self._source_distribution(df),
                "formats": formats,
            },
            "table": df.to_dict(orient="records"),
            "file_paths": file_paths,
        }

    def _flatten_records(self, multi_source_data: Any) -> list[dict]:
        if isinstance(multi_source_data, list):
            return [item for item in multi_source_data if isinstance(item, dict)]

        if isinstance(multi_source_data, dict):
            if all(isinstance(v, list) for v in multi_source_data.values()):
                flattened = []
                for source_type, items in multi_source_data.items():
                    for item in items:
                        if isinstance(item, dict):
                            obj = dict(item)
                            obj.setdefault("source_type", source_type)
                            flattened.append(obj)
                return flattened
            return [multi_source_data]

        return []

    def _normalize_record(self, record: dict) -> dict:
        source_type = self._normalize_source_type(record.get("source_type") or record.get("type") or "web")

        normalized = {
            "记录ID": self._first_non_empty(record, ["id", "record_id", "uid", "doc_id"]),
            "数据源类型": source_type,
            "标题": self._first_non_empty(record, ["title", "标题", "name", "主题"]),
            "来源": self._first_non_empty(record, ["source", "来源", "journal", "venue", "platform", "institution"]),
            "作者与单位": self._normalize_author_org(record),
            "发表时间": self._normalize_date(
                self._first_non_empty(record, ["publish_time", "published_at", "date", "year", "发表时间"])
            ),
            "所属技术领域": self._first_non_empty(record, ["tech_domain", "domain", "field", "所属技术领域"]),
            "技术主题": self._normalize_topic(record),
            "关键问题/目标": self._first_non_empty(record, ["problem", "goal", "objective", "关键问题", "目标"]),
            "核心方法/架构/方案": self._first_non_empty(record, ["method", "architecture", "approach", "核心方法"]),
            "关键效果/指标/成果": self._first_non_empty(record, ["metrics", "results", "performance", "成果"]),
            "创新点": self._first_non_empty(record, ["innovation", "novelty", "创新点"]),
            "局限性/风险/不足": self._first_non_empty(record, ["limitations", "risks", "不足", "局限性"]),
            "潜在应用场景/落地价值": self._first_non_empty(record, ["applications", "use_cases", "value", "应用场景"]),
            "未来演进方向与研究热点": self._first_non_empty(record, ["future_work", "future_trends", "hotspots", "未来方向"]),
        }

        for field in self._all_extension_fields():
            normalized[field] = ""

        extension_values = self._extract_extensions(record, source_type)
        normalized.update(extension_values)

        return normalized

    def _extract_extensions(self, record: dict, source_type: str) -> dict:
        field_aliases = {
            "摘要（中英文）": ["abstract", "摘要", "abstract_cn", "abstract_en"],
            "数据集与实验环境": ["dataset", "datasets", "experiment_env", "实验环境"],
            "与SOTA对比结果": ["sota_comparison", "sota", "benchmark_comparison"],
            "引用量/影响力": ["citation_count", "citations", "impact", "影响力"],
            "实验结论": ["conclusion", "experiment_conclusion", "结论"],
            "技术分类号（IPC/CPC/领域编码）": ["ipc", "cpc", "classification_code", "技术分类号"],
            "权利要求核心点（创新点拆解）": ["claims", "claim_points", "权利要求"],
            "专利价值指标（引用、家族数量、法律状态）": ["patent_value", "family_size", "legal_status", "value_metrics"],
            "项目类别（国家/省部/企业/横向）": ["project_type", "项目类别"],
            "研究目标与考核指标": ["project_objective", "kpi", "研究目标", "考核指标"],
            "技术路线图": ["roadmap", "tech_roadmap", "技术路线图"],
            "成果产出体系（论文、专利、样机、产品）": ["deliverables", "outputs", "成果产出体系"],
            "经费规模/投入强度": ["budget", "funding", "investment", "经费规模"],
            "项目执行阶段状态": ["project_status", "execution_stage", "项目状态"],
            "行业规模与增长率": ["market_size", "growth_rate", "行业规模"],
            "政策驱动因素": ["policy_drivers", "policies", "政策驱动"],
            "关键玩家与生态格局": ["key_players", "ecosystem", "关键玩家"],
            "技术需求、痛点、瓶颈": ["pain_points", "bottlenecks", "tech_needs", "技术痛点"],
            "应用落地节奏": ["adoption_timeline", "commercialization_pace", "落地节奏"],
            "风险评估（技术、市场、政策）": ["risk_assessment", "risks", "风险评估"],
            "关键观点/判断": ["key_viewpoints", "judgement", "观点"],
            "事件类型（发布、会议、突破、争议）": ["event_type", "事件类型"],
            "关联技术/公司/人物": ["related_entities", "related_company", "related_people", "关联技术"],
            "动态情感倾向（正面/负面/中性）": ["sentiment", "情感", "sentiment_label"],
        }

        values = {}
        for field in self.EXT_FIELDS.get(source_type, []):
            val = self._first_non_empty(record, field_aliases.get(field, []))
            if field == "动态情感倾向（正面/负面/中性）":
                val = self._normalize_sentiment(val)
            values[field] = val
        return values

    def _all_extension_fields(self) -> list[str]:
        fields = []
        for _, ext in self.EXT_FIELDS.items():
            fields.extend(ext)
        return fields

    def _normalize_source_type(self, source_type: Any) -> str:
        val = str(source_type).strip().lower()
        for canonical, aliases in self.SOURCE_ALIASES.items():
            if val in aliases:
                return canonical
        return "web"

    def _first_non_empty(self, record: dict, keys: list[str]) -> str:
        for key in keys:
            if key in record and record[key] is not None and str(record[key]).strip() != "":
                return str(record[key]).strip()
        return ""

    def _normalize_author_org(self, record: dict) -> str:
        author = self._first_non_empty(record, ["author", "authors", "作者"])
        org = self._first_non_empty(record, ["affiliation", "organization", "org", "单位"])
        if author and org:
            return f"{author} | {org}"
        return author or org

    def _normalize_topic(self, record: dict) -> str:
        raw = record.get("topics") or record.get("keywords") or record.get("技术主题")
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
            return ", ".join(items[:8])
        if isinstance(raw, str):
            parts = [p.strip() for p in re.split(r"[,，;；|]", raw) if p.strip()]
            return ", ".join(parts[:8])
        return ""

    def _normalize_date(self, value: str) -> str:
        if not value:
            return ""

        value = str(value).strip()
        year_match = re.fullmatch(r"\d{4}", value)
        if year_match:
            return value

        patterns = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%Y-%m",
            "%Y/%m",
            "%Y.%m",
        ]
        for pattern in patterns:
            try:
                dt = datetime.strptime(value, pattern)
                if "%d" in pattern:
                    return dt.strftime("%Y-%m-%d")
                return dt.strftime("%Y-%m")
            except ValueError:
                continue

        match = re.search(r"(19|20)\d{2}", value)
        return match.group(0) if match else value

    def _normalize_sentiment(self, value: str) -> str:
        if not value:
            return ""
        clean = str(value).strip().lower()
        return self.SENTIMENT_ALIASES.get(clean, value)

    def _normalize_output_formats(self, output_formats: list | None) -> list[str]:
        if not output_formats:
            return ["json", "excel", "markdown"]
        allowed = {"json", "excel", "markdown"}
        normalized = [str(fmt).strip().lower() for fmt in output_formats]
        return [fmt for fmt in normalized if fmt in allowed] or ["json", "excel", "markdown"]

    def _to_markdown(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "| 记录ID | 数据源类型 |\n|---|---|\n|  |  |\n"

        columns = list(df.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        rows = []
        for _, row in df.iterrows():
            values = [str(row.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in columns]
            rows.append("| " + " | ".join(values) + " |")

        return "\n".join([header, separator] + rows) + "\n"

    def _source_distribution(self, df: pd.DataFrame) -> dict:
        if "数据源类型" not in df.columns or df.empty:
            return {}
        counts = df["数据源类型"].value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}
    def _llm_enrich_empty_fields(self, row: dict, current_idx: int, total: int) -> dict:
        """调用大模型，根据标题、摘要等上下文补充缺少的关键字段"""
        
        # 定义我们需要大模型重点补全的字段
        target_fields = [
            "关键问题/目标", 
            "核心方法/架构/方案", 
            "关键效果/指标/成果", 
            "创新点", 
            "潜在应用场景/落地价值"
        ]
        
        # 筛选出当前记录中确实为空的字段
        empty_fields = [f for f in target_fields if not row.get(f)]
        if not empty_fields:
            return row  # 都不为空则直接跳过

        # 收集上下文信息（融合论文摘要、专利摘要、新闻正文等）
        context_parts = []
        if row.get("标题"): context_parts.append(f"标题: {row.get('标题')}")
        if row.get("摘要（中英文）"): context_parts.append(f"摘要: {row.get('摘要（中英文）')}")
        if row.get("关键观点/判断"): context_parts.append(f"核心正文: {row.get('关键观点/判断')}")
        if row.get("权利要求核心点（创新点拆解）"): context_parts.append(f"权利要求: {row.get('权利要求核心点（创新点拆解）')}")
        
        context_str = "\n".join(context_parts)
        if len(context_str.strip()) < 20:  
            # 如果没有实质性文本可供分析，直接返回
            return row

        prompt = (
            f"你是专业的技术情报分析专家。请仔细阅读以下资料卡片，提取并精简补全缺失的字段信息。\n"
            f"【材料信息】:\n{context_str}\n\n"
            f"【你需要补全的空字段为】: {', '.join(empty_fields)}\n"
            f"要求:\n"
            f"1. 仅输出合法的 JSON 对象，键必须是上述要求的空字段名。\n"
            f"2. 每个字段值请用 1-2 句话概括，高度专业、精炼。\n"
            f"3. 如果材料中完全未提及某字段的信息，该字段值请填入空字符串 \"\"。\n"
        )

        try:
            # 引入单例大模型
            model = common_utils.ModelProvider.get_model()
            # 根据 smolagents API 调用模型
            response = model([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, "content") else str(response)
            
            # 使用 common_utils 中鲁棒的 JSON 解析器处理 Markdown 代码块及脏数据
            parsed_data = common_utils.safe_json_parse(content, fallback_type=dict)
            
            if isinstance(parsed_data, dict):
                for field in empty_fields:
                    if field in parsed_data and str(parsed_data[field]).strip() and parsed_data[field] != "无":
                        row[field] = str(parsed_data[field]).strip()
                        
            print(f"  └─ 成功补全记录 [{current_idx}/{total}] : {row.get('标题', '')[:15]}...")
            
        except Exception as e:
            # 如果失败则吃掉异常不影响主流程，返回原记录
            print(f"  └─ 补全记录 [{current_idx}/{total}] 时模型解析失败: {str(e)}")
            
        return row

class KeyFrontierTechnologyMiningTool(Tool):
    name = "tool_key_frontier_technology_mining"
    description = (
        "关键前沿技术挖掘工具。以结构化数据表为输入，自动完成技术聚类、前沿识别、内涵提取，"
        "输出关键前沿技术清单与专报。"
    )
    inputs = {
        "structured_table": {
            "type": "any",
            "description": "碎片化信息聚合工具输出的结构化数据表。可传 list[dict] 或包含 table 字段的 dict。",
        },
        "domain_keywords": {
            "type": "array",
            "description": "用户指定技术领域关键词列表，作为聚类和筛选辅助条件。",
            "nullable": True,
        },
        "output_format": {
            "type": "string",
            "description": "返回格式，可选 dict/json/markdown。默认 dict。",
            "nullable": True,
        },
        "min_criteria": {
            "type": "integer",
            "description": "前沿技术判定最少满足条数，默认 2。",
            "nullable": True,
        },
        "out_dir": {
            "type": "string",
            "description": "可选输出目录。设置后自动落盘 json/markdown。",
            "nullable": True,
        },
        "file_stem": {
            "type": "string",
            "description": "输出文件名前缀，默认 key_frontier_technology_report。",
            "nullable": True,
        },
    }
    output_type = "any"

    def forward(
        self,
        structured_table,
        domain_keywords: list | None = None,
        output_format: str | None = None,
        min_criteria: int = 2,
        out_dir: str | None = None,
        file_stem: str | None = None,
    ):
        df = self._to_dataframe(structured_table)
        if df.empty:
            result = {
                "summary": {"total_records": 0, "total_clusters": 0, "frontier_count": 0},
                "frontier_technologies": [],
                "clusters": [],
                "report_markdown": "# 关键前沿技术清单专报\n\n无可分析数据。\n",
            }
            return self._format_result(result, output_format)

        records = df.to_dict(orient="records")
        keyword_set = {str(k).strip() for k in (domain_keywords or []) if str(k).strip()}

        clusters = self._cluster_records(records, keyword_set)
        frontier_items = []
        cluster_outputs = []

        for cluster_id, cluster_records in enumerate(clusters, start=1):
            cluster_profile = self._build_cluster_profile(cluster_id, cluster_records)
            rules = self._evaluate_frontier_rules(cluster_records, min_criteria)
            cluster_profile["frontier_rules"] = rules

            if rules["is_frontier"]:
                elements = self._extract_frontier_elements(cluster_profile, cluster_records)
                item = {
                    "technology_id": f"frontier_{cluster_id:03d}",
                    "technology_name": cluster_profile["technology_name"],
                    "supporting_record_count": len(cluster_records),
                    "supporting_sources": cluster_profile["sources"],
                    "supporting_orgs": cluster_profile["institutions"],
                    "time_trend": cluster_profile["time_trend"],
                    "frontier_rules": rules,
                    "analysis": elements,
                }
                frontier_items.append(item)

            cluster_outputs.append(cluster_profile)

        report_markdown = self._build_report(frontier_items)
        result = {
            "summary": {
                "total_records": len(records),
                "total_clusters": len(clusters),
                "frontier_count": len(frontier_items),
                "domain_keywords": list(keyword_set),
            },
            "frontier_technologies": frontier_items,
            "clusters": cluster_outputs,
            "report_markdown": report_markdown,
        }

        file_paths = self._save_outputs(result, out_dir, file_stem)
        if file_paths:
            result["file_paths"] = file_paths

        return self._format_result(result, output_format)

    def _to_dataframe(self, structured_table: Any) -> pd.DataFrame:
        if isinstance(structured_table, pd.DataFrame):
            return structured_table.copy()

        if isinstance(structured_table, dict):
            if "table" in structured_table and isinstance(structured_table["table"], list):
                return pd.DataFrame(structured_table["table"])
            if "data" in structured_table and isinstance(structured_table["data"], list):
                return pd.DataFrame(structured_table["data"])
            if "records" in structured_table and isinstance(structured_table["records"], list):
                return pd.DataFrame(structured_table["records"])
            if all(not isinstance(v, (list, dict)) for v in structured_table.values()):
                return pd.DataFrame([structured_table])

        if isinstance(structured_table, list):
            return pd.DataFrame([x for x in structured_table if isinstance(x, dict)])

        return pd.DataFrame([])

    def _cluster_records(self, records: list[dict], keyword_set: set[str]) -> list[list[dict]]:
        """升级版：基于 BGE-M3 语义向量与 HDBSCAN 的真实学术聚类"""
        if not records:
            return []

        texts_to_embed = []
        for rec in records:
            text_parts = [
                str(rec.get("技术主题", "")),
                str(rec.get("所属技术领域", "")),
                str(rec.get("关键问题/目标", "")),
                str(rec.get("核心方法/架构/方案", "")),
                str(rec.get("标题", "")),
            ]
            texts_to_embed.append(" ".join(text_parts))

        embedder = BGEM3Embedder(batch_size=64)
        min_size = 2 if len(records) < 30 else 3
        clusterer = UmapHdbscanClusterer(min_cluster_size=min_size)

        print(f"🧠 正在进行高维语义向量化与聚类 (共 {len(records)} 条数据)...")
        vectors = embedder.embed(texts_to_embed)
        labels = clusterer.fit_predict(vectors)

        clusters_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                clusters_dict[label].append(records[idx])
            else:
                pass

        # 修复 1：严格化，找不到关联数据直接丢弃，不强行打包缝合
        if not clusters_dict:
            print("⚠️ 聚类算法未发现明显密度簇，舍弃这些孤立且无关联的数据。")
            return []

        return list(clusters_dict.values())

    def _record_tokens(self, record: dict, keyword_set: set[str]) -> set[str]:
        text_parts = [
            str(record.get("技术主题", "")),
            str(record.get("所属技术领域", "")),
            str(record.get("关键问题/目标", "")),
            str(record.get("核心方法/架构/方案", "")),
            str(record.get("标题", "")),
        ]
        text = " ".join(text_parts)
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", text))
        if keyword_set:
            for kw in keyword_set:
                if kw and kw in text:
                    tokens.add(kw)
        return tokens

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _build_cluster_profile(self, cluster_id: int, records: list[dict]) -> dict:
        topic_counter = Counter()
        sources = set()
        institutions = set()
        years = []

        for rec in records:
            topic_counter.update(self._split_topics(rec.get("技术主题", "")))
            src = str(rec.get("来源", "")).strip()
            if src:
                sources.add(src)
            org = str(rec.get("作者与单位", "")).strip()
            if org:
                institutions.add(org)
            y = self._extract_year(rec.get("发表时间", ""))
            if y:
                years.append(y)

        top_topics = [w for w, _ in topic_counter.most_common(5)]
        tech_name = " / ".join(top_topics[:2]) if top_topics else f"技术簇_{cluster_id:03d}"

        year_count = dict(sorted(Counter(years).items()))
        return {
            "cluster_id": cluster_id,
            "technology_name": tech_name,
            "size": len(records),
            "top_topics": top_topics,
            "sources": sorted(sources),
            "institutions": sorted(institutions),
            "time_trend": year_count,
        }

    def _evaluate_frontier_rules(self, records: list[dict], min_criteria: int) -> dict:
        size = len(records)

        method_metric_ready = 0
        for rec in records:
            if str(rec.get("核心方法/架构/方案", "")).strip() and str(rec.get("关键效果/指标/成果", "")).strip():
                method_metric_ready += 1

        years = sorted([y for y in [self._extract_year(r.get("发表时间", "")) for r in records] if y])
        year_count = Counter(years)
        recent_growth = False
        if year_count:
            max_year = max(year_count)
            recent = year_count.get(max_year, 0) + year_count.get(max_year - 1, 0)
            past = year_count.get(max_year - 2, 0) + year_count.get(max_year - 3, 0)
            recent_growth = recent > past and recent >= 2

        application_ready = any(str(rec.get("潜在应用场景/落地价值", "")).strip() for rec in records)

        institutions = {
            str(rec.get("作者与单位", "")).strip()
            for rec in records
            if str(rec.get("作者与单位", "")).strip()
        }

        rules = {
            "rule_1_cluster_size": size >= 3,
            "rule_2_definition_method_metric": (method_metric_ready / max(size, 1)) >= 0.4,
            "rule_3_recent_growth": recent_growth,
            "rule_4_application_value": application_ready,
            "rule_5_multi_institution": len(institutions) >= 2,
        }

        hit_rules = [k for k, v in rules.items() if v]
        return {
            "rules": rules,
            "hit_count": len(hit_rules),
            "hit_rules": hit_rules,
            "min_required": max(2, int(min_criteria or 2)),
            "is_frontier": len(hit_rules) >= max(2, int(min_criteria or 2)),
        }

    def _extract_frontier_elements(self, cluster_profile: dict, records: list[dict]) -> dict:
        # 删除不合理的 LLM 降级：仅依赖大模型提取，失败则清晰标识，不使用生硬规则污染结果
        llm_result = self._llm_extract(cluster_profile, records)
        if isinstance(llm_result, dict):
            return llm_result
        return {
            "基本定义": "LLM分析失败",
            "拟解决痛点/科学问题": "LLM分析失败",
            "核心原理": "LLM分析失败",
            "参数指标": "LLM分析失败",
            "作用价值": "LLM分析失败",
        }

    def _llm_extract(self, cluster_profile: dict, records: list[dict]) -> dict | None:
        try:
            model = common_utils.ModelProvider.get_model()
        except Exception:
            return None

        sample_lines = []
        for rec in records[:8]:
            sample_lines.append(
                f"标题: {rec.get('标题', '')}\n"
                f"关键问题: {rec.get('关键问题/目标', '')}\n"
                f"核心方法: {rec.get('核心方法/架构/方案', '')}\n"
                f"关键指标: {rec.get('关键效果/指标/成果', '')}\n"
                f"应用价值: {rec.get('潜在应用场景/落地价值', '')}\n"
            )

        prompt = (
            "你是技术情报分析专家。请基于输入材料提取关键前沿技术内涵。\n"
            "输出必须是一个合法 JSON 对象，包含以下键：\n"
            "基本定义、拟解决痛点/科学问题、核心原理、参数指标、作用价值。\n"
            "每个字段要求 1-3 句、简洁客观。\n\n"
            f"技术名称: {cluster_profile.get('technology_name', '')}\n"
            f"技术主题: {', '.join(cluster_profile.get('top_topics', []))}\n"
            "样本材料:\n"
            + "\n---\n".join(sample_lines)
        )

        try:
            response = model([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_json_like(content)
        except Exception:
            return None

    def _parse_json_like(self, text: str) -> dict | None:
        if not isinstance(text, str):
            return None
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            data = json.loads(clean)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = ast.literal_eval(match.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _build_report(self, frontier_items: list[dict]) -> str:
        lines = ["# 关键前沿技术清单专报", ""]
        if not frontier_items:
            lines.append("未识别到满足标准（至少命中2条规则）的关键前沿技术。")
            lines.append("")
            return "\n".join(lines)

        for idx, item in enumerate(frontier_items, start=1):
            ana = item.get("analysis", {})
            rule_meta = item.get("frontier_rules", {})
            lines.append(f"## {idx}. {item.get('technology_name', '')}")
            lines.append("")
            lines.append(f"- 技术ID: {item.get('technology_id', '')}")
            lines.append(f"- 支撑文献/样本数: {item.get('supporting_record_count', 0)}")
            lines.append(f"- 主要来源: {', '.join(item.get('supporting_sources', []))}")
            lines.append(f"- 主要机构: {', '.join(item.get('supporting_orgs', []))}")
            lines.append(f"- 命中识别标准: {rule_meta.get('hit_count', 0)} / 5")
            lines.append(f"- 命中项: {', '.join(rule_meta.get('hit_rules', []))}")
            lines.append("")
            lines.append("### 基本定义")
            lines.append(ana.get("基本定义", ""))
            lines.append("")
            lines.append("### 拟解决痛点/科学问题")
            lines.append(ana.get("拟解决痛点/科学问题", ""))
            lines.append("")
            lines.append("### 核心原理")
            lines.append(ana.get("核心原理", ""))
            lines.append("")
            lines.append("### 参数指标")
            lines.append(ana.get("参数指标", ""))
            lines.append("")
            lines.append("### 作用价值")
            lines.append(ana.get("作用价值", ""))
            lines.append("")

        return "\n".join(lines)

    def _save_outputs(self, result: dict, out_dir: str | None, file_stem: str | None) -> dict:
        if not out_dir:
            return {}

        os.makedirs(out_dir, exist_ok=True)
        stem = file_stem or "key_frontier_technology_report"
        json_path = os.path.join(out_dir, f"{stem}.json")
        md_path = os.path.join(out_dir, f"{stem}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.get("report_markdown", ""))

        return {"json": json_path, "markdown": md_path}

    def _format_result(self, result: dict, output_format: str | None):
        fmt = str(output_format or "dict").lower().strip()
        if fmt == "json":
            return json.dumps(result, ensure_ascii=False, indent=2)
        if fmt == "markdown":
            return result.get("report_markdown", "")
        return result

    def _split_topics(self, topic_text: str) -> list[str]:
        if not topic_text:
            return []
        parts = re.split(r"[,，;；|/]+", str(topic_text))
        return [p.strip() for p in parts if p.strip()]

    def _extract_year(self, date_text: Any) -> int | None:
        if not date_text:
            return None
        match = re.search(r"(19|20)\d{2}", str(date_text))
        if not match:
            return None
        try:
            return int(match.group(0))
        except Exception:
            return None


class WeakSignalTechnologyMiningTool(KeyFrontierTechnologyMiningTool):
    name = "tool_weak_signal_technology_mining"
    description = (
        "弱信号技术识别工具。以结构化数据表为输入，自动完成聚类、弱信号遴选、内涵提取与专报生成，"
        "输出弱信号技术清单。"
    )
    inputs = {
        "structured_table": {
            "type": "any",
            "description": "碎片化信息聚合工具输出的结构化数据表。可传 list[dict] 或包含 table 字段的 dict。",
        },
        "domain_keywords": {
            "type": "array",
            "description": "用户指定技术领域关键词列表，作为聚类和筛选辅助条件。",
            "nullable": True,
        },
        "output_format": {
            "type": "string",
            "description": "返回格式，可选 dict/json/markdown。默认 dict。",
            "nullable": True,
        },
        "min_criteria": {
            "type": "integer",
            "description": "弱信号判定最少满足条数，默认 2。",
            "nullable": True,
        },
        "out_dir": {
            "type": "string",
            "description": "可选输出目录。设置后自动落盘 json/markdown。",
            "nullable": True,
        },
        "file_stem": {
            "type": "string",
            "description": "输出文件名前缀，默认 weak_signal_technology_report。",
            "nullable": True,
        },
    }
    output_type = "any"

    def forward(
        self,
        structured_table,
        domain_keywords: list | None = None,
        output_format: str | None = None,
        min_criteria: int = 2,
        out_dir: str | None = None,
        file_stem: str | None = None,
    ):
        df = self._to_dataframe(structured_table)
        if df.empty:
            result = {
                "summary": {"total_records": 0, "total_clusters": 0, "weak_signal_count": 0},
                "weak_signal_technologies": [],
                "clusters": [],
                "report_markdown": "# 弱信号技术清单专报\n\n无可分析数据。\n",
            }
            return self._format_result(result, output_format)

        records = df.to_dict(orient="records")
        keyword_set = {str(k).strip() for k in (domain_keywords or []) if str(k).strip()}

        clusters = self._cluster_records(records, keyword_set)
        weak_items = []
        cluster_outputs = []

        for cluster_id, cluster_records in enumerate(clusters, start=1):
            cluster_profile = self._build_cluster_profile(cluster_id, cluster_records)
            rules = self._evaluate_weak_signal_rules(cluster_records, min_criteria)
            cluster_profile["weak_signal_rules"] = rules

            if rules["is_weak_signal"]:
                elements = self._extract_frontier_elements(cluster_profile, cluster_records)
                item = {
                    "technology_id": f"weak_{cluster_id:03d}",
                    "technology_name": cluster_profile["technology_name"],
                    "supporting_record_count": len(cluster_records),
                    "supporting_sources": cluster_profile["sources"],
                    "supporting_orgs": cluster_profile["institutions"],
                    "time_trend": cluster_profile["time_trend"],
                    "weak_signal_rules": rules,
                    "analysis": elements,
                }
                weak_items.append(item)

            cluster_outputs.append(cluster_profile)

        report_markdown = self._build_weak_signal_report(weak_items)
        result = {
            "summary": {
                "total_records": len(records),
                "total_clusters": len(clusters),
                "weak_signal_count": len(weak_items),
                "domain_keywords": list(keyword_set),
            },
            "weak_signal_technologies": weak_items,
            "clusters": cluster_outputs,
            "report_markdown": report_markdown,
        }

        file_paths = self._save_weak_outputs(result, out_dir, file_stem)
        if file_paths:
            result["file_paths"] = file_paths

        return self._format_result(result, output_format)

    def _evaluate_weak_signal_rules(self, records: list[dict], min_criteria: int) -> dict:
        size = len(records)

        # 规则1: 研究文献数量极少，尚未形成稳定技术簇
        rule_1_few_records = size <= 2

        # 规则2: 以概念探索/初步验证为主，技术路线不明确
        early_stage_count = 0
        for rec in records:
            method_text = str(rec.get("核心方法/架构/方案", "")).strip()
            metric_text = str(rec.get("关键效果/指标/成果", "")).strip()
            if (not method_text) or (not metric_text):
                early_stage_count += 1
            else:
                concept_hint = str(rec.get("创新点", "")) + " " + str(rec.get("关键问题/目标", ""))
                if re.search(r"探索|初步|概念|原型|可行性|验证", concept_hint):
                    early_stage_count += 1

        rule_2_early_stage = (early_stage_count / max(size, 1)) >= 0.5

        # 规则3: 暂无明显研究热度，未被主流关注
        years = sorted([y for y in [self._extract_year(r.get("发表时间", "")) for r in records] if y])
        year_count = Counter(years)
        no_hotness = True
        if year_count:
            max_year = max(year_count)
            recent = year_count.get(max_year, 0) + year_count.get(max_year - 1, 0)
            past = year_count.get(max_year - 2, 0) + year_count.get(max_year - 3, 0)
            no_hotness = recent <= past

        source_count = len({str(r.get("来源", "")).strip() for r in records if str(r.get("来源", "")).strip()})
        rule_3_low_attention = no_hotness and source_count <= 2

        # 规则4: 新原理/新交叉/新方向，未来潜力待验证
        novel_count = 0
        for rec in records:
            novel_text = " ".join([
                str(rec.get("创新点", "")),
                str(rec.get("所属技术领域", "")),
                str(rec.get("技术主题", "")),
                str(rec.get("未来演进方向与研究热点", "")),
            ])
            if re.search(r"新|首次|首创|交叉|跨学科|融合|新方向|新范式|潜力", novel_text):
                novel_count += 1

        rule_4_novel_potential = (novel_count / max(size, 1)) >= 0.4

        rules = {
            "rule_1_few_records": rule_1_few_records,
            "rule_2_early_stage_unclear_route": rule_2_early_stage,
            "rule_3_low_attention": rule_3_low_attention,
            "rule_4_novel_cross_potential": rule_4_novel_potential,
        }

        hit_rules = [k for k, v in rules.items() if v]
        threshold = max(2, int(min_criteria or 2))
        return {
            "rules": rules,
            "hit_count": len(hit_rules),
            "hit_rules": hit_rules,
            "min_required": threshold,
            "is_weak_signal": len(hit_rules) >= threshold,
        }

    def _build_weak_signal_report(self, weak_items: list[dict]) -> str:
        lines = ["# 弱信号技术清单专报", ""]
        if not weak_items:
            lines.append("未识别到满足标准（至少命中2条规则）的弱信号技术。")
            lines.append("")
            return "\n".join(lines)

        for idx, item in enumerate(weak_items, start=1):
            ana = item.get("analysis", {})
            rule_meta = item.get("weak_signal_rules", {})
            lines.append(f"## {idx}. {item.get('technology_name', '')}")
            lines.append("")
            lines.append(f"- 技术ID: {item.get('technology_id', '')}")
            lines.append(f"- 支撑文献/样本数: {item.get('supporting_record_count', 0)}")
            lines.append(f"- 主要来源: {', '.join(item.get('supporting_sources', []))}")
            lines.append(f"- 主要机构: {', '.join(item.get('supporting_orgs', []))}")
            lines.append(f"- 命中识别标准: {rule_meta.get('hit_count', 0)} / 4")
            lines.append(f"- 命中项: {', '.join(rule_meta.get('hit_rules', []))}")
            lines.append("")
            lines.append("### 基本定义")
            lines.append(ana.get("基本定义", ""))
            lines.append("")
            lines.append("### 拟解决痛点/科学问题")
            lines.append(ana.get("拟解决痛点/科学问题", ""))
            lines.append("")
            lines.append("### 核心原理")
            lines.append(ana.get("核心原理", ""))
            lines.append("")
            lines.append("### 参数指标")
            lines.append(ana.get("参数指标", ""))
            lines.append("")
            lines.append("### 作用价值")
            lines.append(ana.get("作用价值", ""))
            lines.append("")

        return "\n".join(lines)

    def _save_weak_outputs(self, result: dict, out_dir: str | None, file_stem: str | None) -> dict:
        if not out_dir:
            return {}

        os.makedirs(out_dir, exist_ok=True)
        stem = file_stem or "weak_signal_technology_report"
        json_path = os.path.join(out_dir, f"{stem}.json")
        md_path = os.path.join(out_dir, f"{stem}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.get("report_markdown", ""))

        return {"json": json_path, "markdown": md_path}


class EvolutionPathAnalysisTool(KeyFrontierTechnologyMiningTool):
    name = "tool_evolution_path_analysis"
    description = (
        "演进路径分析工具。融合关键前沿技术清单、弱信号清单与原始结构化表，"
        "抽取主流技术路线、实现方案、难题瓶颈及优缺点，并生成结构化专报。"
    )
    inputs = {
        "frontier_list": {
            "type": "any",
            "description": "关键前沿技术挖掘工具输出结果。支持完整 dict 或其中的列表。",
        },
        "weak_signal_list": {
            "type": "any",
            "description": "弱信号技术识别工具输出结果。支持完整 dict 或其中的列表。",
            "nullable": True,
        },
        "structured_table": {
            "type": "any",
            "description": "碎片化信息聚合输出的结构化数据表。",
        },
        "output_format": {
            "type": "string",
            "description": "返回格式，可选 dict/json/markdown。默认 dict。",
            "nullable": True,
        },
        "out_dir": {
            "type": "string",
            "description": "可选输出目录。设置后自动落盘 json/markdown。",
            "nullable": True,
        },
        "file_stem": {
            "type": "string",
            "description": "输出文件名前缀，默认 technology_evolution_report。",
            "nullable": True,
        },
    }
    output_type = "any"

    ROUTE_SYNONYMS = {
        "多智能体小模型协同": ["小模型协同", "多智能体协作", "分布式小模型", "multi-agent", "agent协同"],
        "检索增强生成": ["rag", "检索增强", "retrieval-augmented generation", "知识检索增强"],
        "模型压缩与蒸馏": ["蒸馏", "distillation", "剪枝", "量化", "压缩"],
        "多模态融合": ["multimodal", "多模态", "跨模态", "视觉语言融合"],
        "强化学习对齐": ["rlhf", "强化学习对齐", "偏好优化", "dpo"],
    }

    def forward(
        self,
        frontier_list,
        structured_table,
        weak_signal_list=None,
        output_format: str | None = None,
        out_dir: str | None = None,
        file_stem: str | None = None,
    ):
        df = self._to_dataframe(structured_table)
        frontier_items = self._extract_technology_items(frontier_list, "frontier")
        weak_items = self._extract_technology_items(weak_signal_list, "weak")

        if df.empty and not frontier_items and not weak_items:
            empty_result = {
                "summary": {
                    "total_records": 0,
                    "frontier_count": 0,
                    "weak_signal_count": 0,
                    "mainstream_route_count": 0,
                },
                "technical_overview": "无可分析数据。",
                "mainstream_routes": [],
                "bottlenecks": [],
                "report_markdown": "# 技术演进路径分析专报\n\n无可分析数据。\n",
            }
            return self._format_result(empty_result, output_format)

        records = df.to_dict(orient="records")

        # 步骤2: 要素抽取
        elements = self._extract_route_elements(records, frontier_items, weak_items)

        # 步骤3: 多源信息融合(去重 + 归一 + 合并)
        fused = self._fuse_route_elements(elements)

        # 步骤4: 大模型深度整合与结构化输出
        deep_analysis = self._llm_summarize_evolution(fused, frontier_items, weak_items)

        mainstream_routes = self._build_mainstream_routes(fused)
        
        # 修复 2：严格化，提取出几条就是几条，不再无意义凑数
        mainstream_routes = mainstream_routes[:4]

        result = {
            "summary": {
                "total_records": len(records),
                "frontier_count": len(frontier_items),
                "weak_signal_count": len(weak_items),
                "mainstream_route_count": len(mainstream_routes),
            },
            "technical_overview": deep_analysis.get("technical_overview") or "分析失败（大模型调用异常）",
            "mainstream_routes": mainstream_routes,
            "bottlenecks": self._top_bottlenecks(fused),
            "route_raw_fusion": fused,
        }

        report_markdown = self._build_evolution_report(result)
        result["report_markdown"] = report_markdown

        file_paths = self._save_evolution_outputs(result, out_dir, file_stem)
        if file_paths:
            result["file_paths"] = file_paths

        return self._format_result(result, output_format)

    def _extract_technology_items(self, payload, kind: str) -> list[dict]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            if kind == "frontier" and isinstance(payload.get("frontier_technologies"), list):
                return [x for x in payload["frontier_technologies"] if isinstance(x, dict)]
            if kind == "weak" and isinstance(payload.get("weak_signal_technologies"), list):
                return [x for x in payload["weak_signal_technologies"] if isinstance(x, dict)]
            if isinstance(payload.get("items"), list):
                return [x for x in payload["items"] if isinstance(x, dict)]
            return [payload]
        return []

    def _extract_route_elements(self, records: list[dict], frontier_items: list[dict], weak_items: list[dict]) -> list[dict]:
        elements = []

        for rec in records:
            topics = self._split_topics(str(rec.get("技术主题", "")))
            route_name = topics[0] if topics else str(rec.get("所属技术领域", "")).strip()
            
            # ---- 新增拦截逻辑：如果没名字，就从“核心方法”或“标题”里硬抓一个 ----
            if not route_name:
                method_str = str(rec.get("核心方法/架构/方案", "")).strip()
                if method_str:
                    # 简单提取英文缩写或前几个字作为临时特征名，避免变成毫无意义的“未命名路线”
                    route_name = method_str.split("，")[0].split(",")[0][:12] + "等相关技术"
                else:
                    route_name = "未分类技术路线"

            elements.append({
                "route": route_name,
                "source_type": str(rec.get("数据源类型", "")).strip() or "unknown",
                "implementation": str(rec.get("核心方法/架构/方案", "")).strip(),
                "bottleneck": str(rec.get("局限性/风险/不足", "")).strip() or str(rec.get("关键问题/目标", "")).strip(),
                "advantage": str(rec.get("关键效果/指标/成果", "")).strip(),
                "disadvantage": str(rec.get("局限性/风险/不足", "")).strip(),
                "improvement": str(rec.get("创新点", "")).strip(),
            })

        for item in frontier_items + weak_items:
            analysis = item.get("analysis", {}) if isinstance(item, dict) else {}
            route_name = str(item.get("technology_name", "")).strip() or "未命名路线"
            elements.append({
                "route": route_name,
                "source_type": "technology_list",
                "implementation": str(analysis.get("核心原理", "")).strip(),
                "bottleneck": str(analysis.get("拟解决痛点/科学问题", "")).strip(),
                "advantage": str(analysis.get("作用价值", "")).strip(),
                "disadvantage": "",
                "improvement": str(analysis.get("参数指标", "")).strip(),
            })

        return elements

    def _fuse_route_elements(self, elements: list[dict]) -> dict[str, dict]:
        fused = {}
        for ele in elements:
            raw_route = str(ele.get("route", "")).strip()
            if not raw_route:
                continue
            route = self._normalize_route_name(raw_route)

            if route not in fused:
                fused[route] = {
                    "route": route,
                    "aliases": set(),
                    "source_types": set(),
                    "implementations": [],
                    "bottlenecks": [],
                    "advantages": [],
                    "disadvantages": [],
                    "improvements": [],
                }

            item = fused[route]
            item["aliases"].add(raw_route)
            if ele.get("source_type"):
                item["source_types"].add(ele["source_type"])

            for src_key, dst_key in [
                ("implementation", "implementations"),
                ("bottleneck", "bottlenecks"),
                ("advantage", "advantages"),
                ("disadvantage", "disadvantages"),
                ("improvement", "improvements"),
            ]:
                val = str(ele.get(src_key, "")).strip()
                if val:
                    item[dst_key].append(val)

        for route, item in fused.items():
            for key in ["implementations", "bottlenecks", "advantages", "disadvantages", "improvements"]:
                item[key] = self._dedup_texts(item[key], top_k=8)
            item["aliases"] = sorted(item["aliases"])
            item["source_types"] = sorted(item["source_types"])

        return fused

    def _normalize_route_name(self, name: str) -> str:
        clean = name.strip()
        low = clean.lower()
        for canonical, aliases in self.ROUTE_SYNONYMS.items():
            for alias in aliases:
                if alias.lower() in low or low in alias.lower():
                    return canonical
        return clean

    def _dedup_texts(self, texts: list[str], top_k: int = 6) -> list[str]:
        seen = []
        for txt in texts:
            val = re.sub(r"\s+", " ", txt).strip()
            if not val:
                continue
            if val not in seen:
                seen.append(val)
        return seen[:top_k]

    def _llm_summarize_evolution(self, fused: dict, frontier_items: list[dict], weak_items: list[dict]) -> dict:
        payload = {
            "route_count": len(fused),
            "routes": [
                {
                    "route": v["route"],
                    "implementations": v["implementations"][:3],
                    "bottlenecks": v["bottlenecks"][:3],
                    "advantages": v["advantages"][:3],
                    "disadvantages": v["disadvantages"][:3],
                }
                for v in fused.values()
            ],
            "frontier_count": len(frontier_items),
            "weak_signal_count": len(weak_items),
        }

        try:
            model = common_utils.ModelProvider.get_model()
        except Exception:
            return {"technical_overview": "分析失败（大模型调用异常）"}

        prompt = (
            "你是技术演进分析专家。请基于输入 JSON 生成技术概述。\n"
            "只输出合法 JSON，包含键 technical_overview。\n"
            "technical_overview 要覆盖: 定义、解决问题、阶段特征、未来走向。\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            resp = model([{"role": "user", "content": prompt}])
            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = self._parse_json_like(content)
            if isinstance(parsed, dict) and parsed.get("technical_overview"):
                return parsed
        except Exception:
            pass

        return {"technical_overview": "分析失败（大模型调用异常）"}

    def _build_mainstream_routes(self, fused: dict[str, dict]) -> list[dict]:
        ranked = sorted(
            fused.values(),
            key=lambda x: (
                len(x.get("source_types", []))
                + len(x.get("implementations", []))
                + len(x.get("advantages", []))
            ),
            reverse=True,
        )

        routes = []
        for item in ranked:
            routes.append({
                "route_name": item["route"],
                "route_aliases": item["aliases"],
                "core_implementations": item["implementations"][:4],
                "bottlenecks": item["bottlenecks"][:4],
                "advantages": item["advantages"][:4],
                "disadvantages": item["disadvantages"][:4],
                "source_types": item["source_types"],
            })
        return routes

    def _top_bottlenecks(self, fused: dict[str, dict]) -> list[str]:
        all_b = []
        for item in fused.values():
            all_b.extend(item.get("bottlenecks", []))
        return self._dedup_texts(all_b, top_k=10)

    def _build_evolution_report(self, result: dict) -> str:
        lines = ["# 技术演进路径分析专报", ""]
        lines.append("## 技术概述")
        lines.append(result.get("technical_overview", ""))
        lines.append("")

        lines.append("## 主流技术路线")
        for idx, route in enumerate(result.get("mainstream_routes", []), start=1):
            lines.append(f"### {idx}. {route.get('route_name', '')}")
            lines.append(f"- 别名归一: {', '.join(route.get('route_aliases', []))}")
            lines.append(f"- 核心实现方案: {'；'.join(route.get('core_implementations', []))}")
            lines.append(f"- 关键难题与瓶颈: {'；'.join(route.get('bottlenecks', []))}")
            lines.append(f"- 路线优势: {'；'.join(route.get('advantages', []))}")
            lines.append(f"- 路线劣势: {'；'.join(route.get('disadvantages', []))}")
            lines.append("")

        lines.append("## 共性瓶颈")
        for b in result.get("bottlenecks", []):
            lines.append(f"- {b}")
        lines.append("")

        return "\n".join(lines)

    def _save_evolution_outputs(self, result: dict, out_dir: str | None, file_stem: str | None) -> dict:
        if not out_dir:
            return {}

        os.makedirs(out_dir, exist_ok=True)
        stem = file_stem or "technology_evolution_report"
        json_path = os.path.join(out_dir, f"{stem}.json")
        md_path = os.path.join(out_dir, f"{stem}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.get("report_markdown", ""))

        return {"json": json_path, "markdown": md_path}


class KeyTechnologyEvolutionSensingTool(EvolutionPathAnalysisTool):
    name = "tool_key_technology_evolution_sensing"
    description = (
        "重点技术演进态势感知工具。融合关键前沿/弱信号清单与原始结构化数据，"
        "识别成熟度阶段、未来发展趋势并生成结构化专题专报。"
    )
    inputs = {
        "frontier_list": {
            "type": "any",
            "description": "关键前沿技术挖掘工具输出结果。支持完整 dict 或其中的列表。",
        },
        "weak_signal_list": {
            "type": "any",
            "description": "弱信号技术识别工具输出结果。支持完整 dict 或其中的列表。",
            "nullable": True,
        },
        "structured_table": {
            "type": "any",
            "description": "碎片化信息聚合输出的结构化数据表。",
        },
        "technology_name": {
            "type": "string",
            "description": "技术名称，用于专题报告命名中的 XXX。",
            "nullable": True,
        },
        "output_format": {
            "type": "string",
            "description": "返回格式，可选 dict/json/markdown。默认 dict。",
            "nullable": True,
        },
        "out_dir": {
            "type": "string",
            "description": "可选输出目录。设置后自动落盘 json/markdown。",
            "nullable": True,
        },
        "file_stem": {
            "type": "string",
            "description": "输出文件名前缀，默认 key_technology_evolution_sensing。",
            "nullable": True,
        },
    }
    output_type = "any"

    def forward(
        self,
        frontier_list,
        structured_table,
        weak_signal_list=None,
        technology_name: str | None = None,
        output_format: str | None = None,
        out_dir: str | None = None,
        file_stem: str | None = None,
    ):
        df = self._to_dataframe(structured_table)
        frontier_items = self._extract_technology_items(frontier_list, "frontier")
        weak_items = self._extract_technology_items(weak_signal_list, "weak")

        if df.empty and not frontier_items and not weak_items:
            empty = {
                "summary": {
                    "total_records": 0,
                    "frontier_count": 0,
                    "weak_signal_count": 0,
                    "route_count": 0,
                },
                "route_sensing": [],
                "overall_conclusion": "无可分析数据。",
                "frontier_special_report": "# 技术的关键前沿技术识别与分析研判\n\n无可分析数据。\n",
                "weak_signal_special_report": "# 技术的弱信号技术识别与分析研判\n\n无可分析数据。\n",
            }
            return self._format_result(empty, output_format)

        records = df.to_dict(orient="records")
        elements = self._extract_route_elements(records, frontier_items, weak_items)
        fused = self._fuse_route_elements(elements)

        route_sensing = []
        for route_name, route_data in fused.items():
            llm_result = self._llm_refine_maturity_and_trend(route_name, route_data)
            
            # 删除不合理的 LLM 降级：仅依赖大模型研判，不再使用生硬代码规则强行拼凑
            if llm_result:
                merged = llm_result
            else:
                merged = {
                    "maturity_stage": "分析失败",
                    "future_trend": "分析失败",
                    "trend_reasoning": "分析失败（大模型调用异常）"
                }

            route_sensing.append({
                "route_name": route_name,
                "route_aliases": route_data.get("aliases", []),
                "source_types": route_data.get("source_types", []),
                "maturity_stage": merged.get("maturity_stage", "分析失败"),
                "future_trend": merged.get("future_trend", "分析失败"),
                "trend_reasoning": merged.get("trend_reasoning", "分析失败"),
                "core_implementations": route_data.get("implementations", [])[:4],
                "improvement_signals": route_data.get("improvements", [])[:4],
            })

        route_sensing = sorted(route_sensing, key=lambda x: self._maturity_rank(x.get("maturity_stage", "")), reverse=True)

        overall_conclusion = self._build_overall_conclusion(route_sensing, frontier_items, weak_items)
        title_base = (technology_name or "该").strip() + "技术"
        frontier_report = self._build_special_report(title_base, "关键前沿", frontier_items, route_sensing)
        weak_report = self._build_special_report(title_base, "弱信号", weak_items, route_sensing)

        result = {
            "summary": {
                "total_records": len(records),
                "frontier_count": len(frontier_items),
                "weak_signal_count": len(weak_items),
                "route_count": len(route_sensing),
            },
            "route_sensing": route_sensing,
            "overall_conclusion": overall_conclusion,
            "frontier_special_report": frontier_report,
            "weak_signal_special_report": weak_report,
        }

        if out_dir:
            paths = self._save_sensing_outputs(result, out_dir, file_stem)
            result["file_paths"] = paths

        if str(output_format or "").lower().strip() == "markdown":
            return frontier_report + "\n\n---\n\n" + weak_report
        return self._format_result(result, output_format)

    def _llm_refine_maturity_and_trend(self, route_name: str, route_data: dict) -> dict | None:
        payload = {
            "route_name": route_name,
            "implementations": route_data.get("implementations", [])[:4],
            "advantages": route_data.get("advantages", [])[:4],
            "bottlenecks": route_data.get("bottlenecks", [])[:4],
            "improvements": route_data.get("improvements", [])[:4],
        }

        try:
            model = common_utils.ModelProvider.get_model()
        except Exception:
            return None

        prompt = (
            "你是极其严谨的技术情报态势分析专家。请根据输入 JSON 判断技术路线的成熟度与趋势。\n"
            "仅输出合法 JSON，键包括：maturity_stage, future_trend, trend_reasoning。\n"
            "【严格约束】:\n"
            "1. maturity_stage 仅允许：早期探索期、规模验证期、工程化应用期。\n"
            "2. future_trend 仅允许：持续增长、瓶颈停滞、技术迭代。\n"
            "3. trend_reasoning (重点)：禁止使用“具备明确工艺路径”、“展现出优势”等空泛套话！你必须直接引用输入数据中的【具体技术名称、具体数值指标、或特定工艺】来作为你的推理依据。字数控制在80字以内。\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            resp = model([{"role": "user", "content": prompt}])
            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = self._parse_json_like(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _maturity_rank(self, stage: str) -> int:
        mapping = {
            "工程化应用期": 3,
            "规模验证期": 2,
            "早期探索期": 1,
        }
        return mapping.get(stage, 0)

    def _build_overall_conclusion(self, route_sensing: list[dict], frontier_items: list[dict], weak_items: list[dict]) -> str:
        if not route_sensing:
            return "当前样本未形成可判读的演进态势。"

        valid_stages = [r.get("maturity_stage") for r in route_sensing if r.get("maturity_stage") != "分析失败"]
        valid_trends = [r.get("future_trend") for r in route_sensing if r.get("future_trend") != "分析失败"]
        
        if not valid_stages or not valid_trends:
            return "由于核心研判失败，当前无法生成有效的宏观演进态势结论。"

        stage_counter = Counter(valid_stages)
        top_stage = stage_counter.most_common(1)[0][0]
        trend_counter = Counter(valid_trends)
        top_trend = trend_counter.most_common(1)[0][0]

        return (
            f"当前技术路线整体以“{top_stage}”为主，未来趋势呈现“{top_trend}”特征。"
            f"其中关键前沿方向 {len(frontier_items)} 个、弱信号方向 {len(weak_items)} 个，"
            "建议采用“主路线持续优化 + 弱信号定向孵化”的双轨布局。"
        )

    def _build_special_report(self, title_base: str, report_type: str, tech_items: list[dict], route_sensing: list[dict]) -> str:
        if report_type == "关键前沿":
            title = f"# 《{title_base}的关键前沿技术识别与分析研判》"
        else:
            title = f"# 《{title_base}的弱信号技术识别与分析研判》"

        lines = [title, ""]

        # 【核心修复 1：空数据直接熔断】
        # 如果传入的技术列表为空，直接返回干净的结论，终止后续瞎编
        if not tech_items:
            lines.append(f"经系统分析，本次采集的数据集中未发现具备明显特征的{report_type}技术。")
            lines.append("")
            return "\n".join(lines)

        # 【核心修复 2：数据隔离墙】
        # 提取属于当前报告的合法技术名称集合，防止前沿技术“串台”到弱信号报告里
        valid_route_names = {item.get("technology_name", "").strip() for item in tech_items}
        
        # 过滤 route_sensing，只保留属于当前列表的路线
        display_sensing = []
        for row in route_sensing:
            r_name = row.get("route_name", "")
            r_aliases = row.get("route_aliases", [])
            # 只要路线的主名或别名在合法集合中，才允许上榜
            if r_name in valid_route_names or any(a in valid_route_names for a in r_aliases):
                display_sensing.append(row)

        lines.append("## 一、技术态势概览")
        lines.append(f"- 技术条目数: {len(tech_items)}")
        lines.append(f"- 路线覆盖数: {len(display_sensing)}")
        lines.append("")

        lines.append("## 二、成熟度阶段判断")
        for row in display_sensing[:6]:
            lines.append(f"- {row.get('route_name', '')}: {row.get('maturity_stage', '')}")
        lines.append("")

        lines.append("## 三、未来发展趋势")
        for row in display_sensing[:6]:
            lines.append(f"- {row.get('route_name', '')}: {row.get('future_trend', '')}（{row.get('trend_reasoning', '')}）")
        lines.append("")

        lines.append("## 四、重点研判结论")
        if report_type == "关键前沿":
            lines.append("- 面向关键前沿方向，建议围绕工程化落地能力、可复现性和标准体系建设持续投入。")
        else:
            lines.append("- 面向弱信号方向，建议小规模试验验证、持续监测热度变化并建立早期预警机制。")
        lines.append("")

        return "\n".join(lines)

    def _save_sensing_outputs(self, result: dict, out_dir: str, file_stem: str | None) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        stem = file_stem or "key_technology_evolution_sensing"

        json_path = os.path.join(out_dir, f"{stem}.json")
        frontier_md_path = os.path.join(out_dir, f"{stem}_frontier.md")
        weak_md_path = os.path.join(out_dir, f"{stem}_weak_signal.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(frontier_md_path, "w", encoding="utf-8") as f:
            f.write(result.get("frontier_special_report", ""))
        with open(weak_md_path, "w", encoding="utf-8") as f:
            f.write(result.get("weak_signal_special_report", ""))

        return {
            "json": json_path,
            "frontier_report_markdown": frontier_md_path,
            "weak_signal_report_markdown": weak_md_path,
        }

class BGEM3Embedder:
    """BGE-M3 向量器：将文本转化为高维语义向量"""
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        arr = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [list(map(float, row)) for row in arr]

class UmapHdbscanClusterer:
    """UMAP降维 + HDBSCAN 密度聚类"""
    def __init__(self, umap_dim: int = 10, min_cluster_size: int = 3):
        self.umap_dim = umap_dim
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, vectors: list[list[float]]) -> list[int]:
        if not vectors:
            return []
        import numpy as np
        import umap
        from hdbscan import HDBSCAN

        x = np.asarray(vectors, dtype=float)
        
        if len(vectors) < 10:
            reduced = x
        else:
            reducer = umap.UMAP(
                n_components=self.umap_dim,
                metric="cosine",
                random_state=42,
                n_jobs=1,
                n_neighbors=min(15, len(vectors) - 1),
                min_dist=0.0,
            )
            reduced = reducer.fit_transform(x)
            
        labels = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean").fit_predict(reduced)
        return [int(lbl) for lbl in labels]