import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
from smolagents import Tool

class AcademicSearchTool(Tool):
    name = "academic_search"
    description = "用于在 ArXiv 或 OpenAlex 上检索学术论文。输入必须是英文关键词。返回包含文献及其摘要的字典。"
    inputs = {
        "search_query": {
            "type": "string", 
            "description": "检索核心英文关键词。如 'large language model reasoning'"
        },
        "engine": {
            "type": "string", 
            "description": "搜索引擎，可选 'arxiv' 或 'openalex'。默认 'arxiv'",
            "nullable": True
        },
        "sort_by": {
            "type": "string",
            "description": "排序方式。可选 'date' (按时间降序) 或 'citation' (按引用量降序，注意：此项仅对 openalex 引擎有效)。默认 'date'",
            "nullable": True
        },
        "year_start": {
            "type": "integer",
            "description": "起始年份，如 2022。若无限制可不传。",
            "nullable": True
        },
        "year_end": {
            "type": "integer",
            "description": "结束年份，如 2024。若无限制可不传。",
            "nullable": True
        },
        "max_results": {
            "type": "integer", 
            "description": "最大返回数量，默认 10",
            "nullable": True
        }
    }
    # 将输出类型从 string 改为 any，允许返回 Python 字典
    output_type = "any"

    def _reconstruct_openalex_abstract(self, inverted_index: dict) -> str:
        """OpenAlex 的摘要是倒排索引格式，需要还原成普通文本"""
        if not inverted_index:
            return "No abstract available."
        try:
            max_idx = max(max([positions for positions in inverted_index.values()], key=lambda x: max(x)))
            words = [""] * (max_idx + 1)
            for word, positions in inverted_index.items():
                for pos in positions:
                    words[pos] = word
            return " ".join(words).strip()
        except Exception:
            return "Abstract parsing error."

    def forward(self, search_query: str, engine: str = "arxiv", sort_by: str = "date", year_start: int = None, year_end: int = None, max_results: int = 10) -> dict:
        engine = (engine or "arxiv").lower()
        sort_by = (sort_by or "date").lower()
        max_results = max_results or 10
        
        current_year = datetime.datetime.now().year
        y_start = year_start or 2000
        y_end = year_end or current_year
        if y_start > y_end: y_start, y_end = y_end, y_start

        # 初始化符合要求的返回字典格式
        result_dict = {
            "available_citations": {},
            "candidate_keywords": []
        }
        
        keywords_set = set()

        try:
            if engine == "openalex":
                encoded_query = urllib.parse.quote(search_query.strip())
                sort_param = "cited_by_count:desc" if sort_by == "citation" else "publication_date:desc"
                url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:{y_start}-{y_end}&sort={sort_param}&per-page={max_results}"
                headers = {'User-Agent': 'mailto:open_deep_research@example.com'}
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    for paper in data.get('results', []):
                        title = paper.get('title')
                        if not title: continue
                            
                        year = str(paper.get('publication_year', 'Unknown'))
                        url_str = paper.get('id', '')
                        
                        # 提取并还原摘要
                        inv_idx = paper.get('abstract_inverted_index')
                        abstract = self._reconstruct_openalex_abstract(inv_idx)
                        
                        # 提取概念作为关键词
                        for concept in paper.get('concepts', []):
                            concept_name = concept.get('display_name')
                            if concept_name:
                                keywords_set.add(concept_name.lower())

                        result_dict["available_citations"][title] = {
                            "title": title,
                            "year": year,
                            "url": url_str,
                            "abstract": abstract
                        }
            else:
                # 默认 ArXiv
                encoded_query = urllib.parse.quote(f'all:{search_query.strip()}')
                url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results * 2}&sortBy=submittedDate&sortOrder=descending"
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    root = ET.fromstring(response.read().decode('utf-8'))
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    for entry in root.findall('atom:entry', ns):
                        if len(result_dict["available_citations"]) >= max_results: break
                        published = entry.find('atom:published', ns).text
                        year_int = int(published.split('-')[0]) if published else current_year
                        
                        # ArXiv 年份过滤
                        if year_int < y_start or year_int > y_end: continue
                            
                        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                        url_str = entry.find('atom:id', ns).text.strip()
                        
                        # 提取摘要
                        summary_elem = entry.find('atom:summary', ns)
                        abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else "No abstract available."

                        result_dict["available_citations"][title] = {
                            "title": title,
                            "year": str(year_int),
                            "url": url_str,
                            "abstract": abstract
                        }

            # 处理 candidate_keywords
            if keywords_set:
                result_dict["candidate_keywords"] = list(keywords_set)[:15]  # 取前15个高质量关键词
            else:
                # Fallback: 如果没有获取到概念，直接切分搜索词作为候选
                fallback_kws = [k.strip() for k in search_query.replace('OR', ' ').replace('AND', ' ').replace('"', '').split() if len(k.strip()) > 3]
                result_dict["candidate_keywords"] = list(set(fallback_kws))[:10]
            
            return result_dict
            
        except Exception as e:
            # 容错兜底：发生任何请求异常（如 SSL 错误），依然返回符合格式的空字典，避免 Agent 解析崩溃
            print(f"[AcademicSearchTool] 检索出现异常: {str(e)}")
            return {"available_citations": {}, "candidate_keywords": []}