import logging
import urllib.request
import urllib.error
import urllib.parse
import json
import datetime
import time
import socket
import xml.etree.ElementTree as ET
from typing import Dict, Any
from smolagents.monitoring import LogLevel
from smolagents import ChatMessage, MessageRole
from utils.common_utils import safe_json_parse
logger = logging.getLogger(__name__)

class AcademicSearchService:
    """
    双引擎学术检索服务：支持 ArXiv (按时间排序) 与 OpenAlex (按引用量/时间排序)。
    内置 3 次抗网络抖动重试机制。
    """

    @staticmethod
    def parse_search_intent(model, user_input: str) -> dict:
        """调用大模型，将用户输入转换为通用的英文关键词"""
        current_year = datetime.datetime.now().year
        prompt = f"""
        请分析用户的检索需求，提取用于学术数据库的检索参数。
        用户输入: "{user_input}"
        
        要求：
        1. 必须将中文意图翻译为最准确的【英文学术关键词】。
        2. search_query: 提取核心英文搜索词。
           🚨【警告】：绝对禁止保留“前沿”、“进展”、“最新”、“现状”、“趋势”、“展望”等修饰性虚词！
           如果用户输入全是虚词，请根据上下文脑补出具体的硬核技术词汇（如 reasoning, RLHF 等）。
           🚨【致命格式警告】：所有的搜索词必须拼成【一整个普通字符串】，绝对禁止在字符串内部使用双引号！
           ✅ 正确写法: "large language model carbon footprint"
           ❌ 错误写法: "large language model" "carbon footprint" (内部多余的双引号会导致系统崩溃！)
        3. year_start / year_end: 提取时间限制，格式为 YYYY。
           💡 当前真实的年份是 {current_year} 年！如果用户要求“近期”、“前沿”、“最新”，强制将 year_end 设为 {current_year}，year_start 设为 {current_year - 2}。
        
        必须输出合法 JSON：
        {{
            "search_query": "\"large language model\" reasoning",
            "year_start": "{current_year - 2}",
            "year_end": "{current_year}"
        }}
        """
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            response = model(messages).content
            parsed = safe_json_parse(str(response))
            logger.info(f"🔍 意图解析结果: {parsed}")
            return parsed
        except Exception as e:
            logger.error(f"⚠️ 意图解析失败: {e}")
            return {"search_query": '"large language model"', "year_start": "", "year_end": ""}

    @staticmethod
    def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
        """还原 OpenAlex 的倒排索引摘要"""
        if not inverted_index: return ""
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda x: x[0])
        return " ".join([w[1] for w in words])

    @staticmethod
    def search_academic_papers(agent, search_query: str, year_start: str = "", year_end: str = "", max_fetch: int = 20, target_count: int = 10) -> Dict[str, Dict[str, Any]]:
        # 【核心配置读取】：从 agent.state 中读取用户的引擎和排序偏好
        engine = agent.state.get("search_engine", "arxiv").lower() # 默认: "arxiv" 或 "openalex"
        sort_by = agent.state.get("search_sort", "date").lower()   # 默认: "date" (时间) 或 "citation" (引用，仅对 OpenAlex 有效)
        
        citations_dict: Dict[str, Dict[str, Any]] = {}
        if not search_query: return citations_dict

        # 年份处理
        current_year = datetime.datetime.now().year
        y_start = 2017
        y_end = current_year
        try:
            if year_start: y_start = int(year_start)
            if year_end: y_end = int(year_end)
            if y_start > y_end: y_start, y_end = y_end, y_start
        except Exception:
            pass

        logger.info(f"🌐 准备发起学术检索 | 引擎: {engine.upper()} | 排序: {sort_by.upper()} | 关键词: {search_query}")

        # 组装请求对象
        req = None
        if engine == "openalex":
            # OpenAlex 检索组装
            encoded_query = urllib.parse.quote(search_query.strip())
            sort_param = "cited_by_count:desc" if sort_by == "citation" else "publication_date:desc"
            url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:{y_start}-{y_end}&sort={sort_param}&per-page={max_fetch}"
            headers = {'User-Agent': 'mailto:open_deep_research@example.com'} # 进 OpenAlex 的 Polite Pool
            req = urllib.request.Request(url, headers=headers)
        else:
            # ArXiv 检索组装 (强制按 submittedDate 降序)
            encoded_query = urllib.parse.quote(f'all:{search_query.strip()}')
            url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_fetch}&sortBy=submittedDate&sortOrder=descending"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)

        # 3次抗抖动重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    if engine == "openalex":
                        # --------- OpenAlex 解析 ---------
                        data = json.loads(response.read().decode('utf-8'))
                        results = data.get('results', [])
                        for paper in results:
                            if len(citations_dict) >= target_count: break
                            title = str(paper.get('title') or "").strip()
                            abstract = AcademicSearchService._reconstruct_openalex_abstract(paper.get('abstract_inverted_index', {}))
                            if not title or not abstract: continue
                            
                            authors = [a.get('author', {}).get('display_name', '') for a in paper.get('authorships', [])]
                            authors = [a for a in authors if a]
                            authors_str = ", ".join(authors) if authors else "Unknown"
                            year = str(paper.get('publication_year') or y_end)
                            citation_count = paper.get('cited_by_count', 0)
                            url_str = paper.get('id', '')
                            
                            apa_citation = ""
                            if authors and year:
                                first_last = authors[0].split()[-1]
                                if len(authors) == 1: apa_citation = f"({first_last}, {year})"
                                elif len(authors) == 2: apa_citation = f"({first_last} & {authors[1].split()[-1]}, {year})"
                                else: apa_citation = f"({first_last} et al., {year})"
                            
                            citations_dict[title] = {
                                "title": title, "abstract": abstract, "authors": authors_str, "url": url_str,
                                "year": year, "citation_count": citation_count, "source_type": "paper", "apa_citation": apa_citation
                            }
                    else:
                        # --------- ArXiv 解析 ---------
                        root = ET.fromstring(response.read().decode('utf-8'))
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        for entry in root.findall('atom:entry', ns):
                            if len(citations_dict) >= target_count: break
                            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                            url_str = entry.find('atom:id', ns).text.strip()
                            published = entry.find('atom:published', ns).text
                            year = published.split('-')[0] if published else str(y_end)
                            
                            # ArXiv 强依赖本地年份过滤
                            if int(year) < y_start or int(year) > y_end: continue

                            author_elements = entry.findall('atom:author/atom:name', ns)
                            authors = [a.text.strip() for a in author_elements if a.text]
                            authors_str = ", ".join(authors) if authors else "Unknown"
                            
                            apa_citation = ""
                            if authors and year:
                                first_last = authors[0].split()[-1]
                                if len(authors) == 1: apa_citation = f"({first_last}, {year})"
                                elif len(authors) == 2: apa_citation = f"({first_last} & {authors[1].split()[-1]}, {year})"
                                else: apa_citation = f"({first_last} et al., {year})"
                            
                            citations_dict[title] = {
                                "title": title, "abstract": abstract, "authors": authors_str, "url": url_str,
                                "year": year, "citation_count": 0, "source_type": "paper", "apa_citation": apa_citation
                            }

                logger.info(f"✅ {engine.upper()} 检索成功！获取到 {len(citations_dict)} 篇有效文献。")
                break 

            except urllib.error.HTTPError as e:
                logger.info(f"⚠️ HTTP 异常 {e.code}: {e.reason} ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1: time.sleep(3)
            except (urllib.error.URLError, socket.timeout, ConnectionResetError) as e:
                logger.info(f"⚠️ 网络波动检测 ({type(e).__name__}: {e}) ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1: time.sleep(2)
            except Exception as e:
                logger.error(f"❌ 检索发生未知错误: {e}")
                break
                
        return citations_dict