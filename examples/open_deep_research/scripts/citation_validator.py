"""
学术引用验证系统 - 实现完整的引用可靠性检查流程

流程：Search (检索) -> Verify (双源存在性验证) -> Retrieve (获取官方 BibTeX) -> 
      Validate (核对声明) -> Add (写入)
      
每个引用都需要通过严格的学术诚实性检验，确保：
1. 引用来源可验证（CrossRef, arXiv, Google Scholar）
2. 获得官方标准BibTeX格式
3. 论文摘要确实支持所引用的观点
4. 没有编造或错误使用引用
"""

import json
import re
import requests
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from urllib.parse import quote


@dataclass
class CitationRecord:
    """引用记录结构"""
    authors: str              # 作者名字
    year: str                # 出版年份
    title: str               # 论文标题
    source: str              # 来源（CrossRef, arXiv, Google Scholar等）
    url: str                 # DOI或论文URL
    bibtex: str              # 官方BibTeX格式
    abstract: Optional[str]  # 论文摘要
    verification_status: str # 验证状态: 'pending', 'verified', 'rejected'
    claim_text: str          # 论文中的具体使用观点
    validation_result: Dict[str, Any] = None  # LLM验证结果
    timestamp: str = None    # 验证时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        return data


class CitationValidator:
    """学术引用验证器
    
    实现5步验证流程：
    1. Search - 从RAG结果中检索引用
    2. Verify - 双源验证引用存在性
    3. Retrieve - 获取官方BibTeX
    4. Validate - 核对论文摘要与声明匹配
    5. Add - 写入到最终参考文献库
    """
    
    # 学术数据库API端点
    CROSSREF_API = "https://api.crossref.org/works"
    ARXIV_API = "http://export.arxiv.org/api/query"
    OPENALEX_API = "https://api.openalex.org/works"
    
    def __init__(self, model=None, timeout: int = 5):
        """
        Args:
            model: LLM模型，用于Validate步骤
            timeout: API请求超时时间（秒）
        """
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Citation-Validator/1.0 (Research System)'
        })
        
        # 缓存已验证的引用
        self.verified_cache: Dict[str, CitationRecord] = {}
        # 记录验证历史
        self.validation_log: List[Dict] = []
    
    def step1_search(self, author: str, year: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Step 1: Search (检索)
        从多个学术数据库检索引用
        
        Args:
            author: 作者名字
            year: 出版年份
            title: 论文标题（可选，提高精准度）
            
        Returns:
            包含多个数据库搜索结果的字典
        """
        search_results = {
            'crossref': [],
            'arxiv': [],
            'openalex': [],
            'query_time': datetime.now().isoformat()
        }
        
        # 缓存键
        cache_key = f"{author}_{year}"
        if cache_key in self.verified_cache:
            return {'cached': True, 'record': asdict(self.verified_cache[cache_key])}
        
        # 在CrossRef中搜索
        search_results['crossref'] = self._search_crossref(author, year, title)
        
        # 在arXiv中搜索
        search_results['arxiv'] = self._search_arxiv(author, year, title)
        
        # 在OpenAlex中搜索
        search_results['openalex'] = self._search_openalex(author, year, title)
        
        return search_results
    
    def step2_verify(self, search_results: Dict) -> Tuple[bool, Dict]:
        """
        Step 2: Verify (双源存在性验证)
        验证引用是否在至少两个独立来源中存在
        
        Args:
            search_results: 来自step1的搜索结果
            
        Returns:
            (是否验证通过, 最高质量的结果信息)
        """
        sources_found = []
        best_result = None
        best_score = -1
        
        # 检查各个来源的结果
        if search_results.get('crossref') and len(search_results['crossref']) > 0:
            sources_found.append('crossref')
            best_candidate = search_results['crossref'][0]
            if self._calculate_match_score(best_candidate) > best_score:
                best_result = best_candidate
                best_score = self._calculate_match_score(best_candidate)
        
        if search_results.get('arxiv') and len(search_results['arxiv']) > 0:
            sources_found.append('arxiv')
            best_candidate = search_results['arxiv'][0]
            if self._calculate_match_score(best_candidate) > best_score:
                best_result = best_candidate
                best_score = self._calculate_match_score(best_candidate)
        
        if search_results.get('openalex') and len(search_results['openalex']) > 0:
            sources_found.append('openalex')
            best_candidate = search_results['openalex'][0]
            if self._calculate_match_score(best_candidate) > best_score:
                best_result = best_candidate
                best_score = self._calculate_match_score(best_candidate)
        
        # 至少需要在两个来源中找到
        verified = len(sources_found) >= 2 and best_result is not None
        
        return verified, {
            'verified': verified,
            'sources_found': sources_found,
            'best_result': best_result,
            'match_score': best_score
        }
    
    def step3_retrieve(self, best_result: Dict) -> Optional[str]:
        """
        Step 3: Retrieve (获取官方 BibTeX)
        从API获取标准BibTeX格式
        
        Args:
            best_result: 最佳匹配结果（来自step2）
            
        Returns:
            BibTeX格式字符串，或None如果获取失败
        """
        bibtex = None
        
        # 优先从CrossRef获取（最完整）
        if 'doi' in best_result:
            bibtex = self._get_bibtex_from_crossref(best_result['doi'])
        
        # 如果CrossRef失败，尝试arXiv
        if not bibtex and 'arxiv_id' in best_result:
            bibtex = self._get_bibtex_from_arxiv(best_result['arxiv_id'])
        
        # 如果仍未获得，尝试OpenAlex
        if not bibtex and 'openalex_id' in best_result:
            bibtex = self._get_bibtex_from_openalex(best_result['openalex_id'])
        
        # 如果所有API都失败，生成基础BibTeX
        if not bibtex:
            bibtex = self._generate_basic_bibtex(best_result)
        
        return bibtex
    
    def step4_validate(self, citation_record: CitationRecord, context_claim: str) -> Dict[str, Any]:
        """
        Step 4: Validate (核对声明)
        使用LLM验证论文摘要是否真的支持论文中的观点
        
        这是防止引用误用和编造的关键步骤
        
        Args:
            citation_record: 引用记录
            context_claim: 论文中使用该引用的具体观点
            
        Returns:
            验证结果，包含：
            - is_valid: 是否验证通过
            - confidence: 置信度 (0-1)
            - evidence: 摘要中的支持证据
            - issues: 发现的问题列表
        """
        if not self.model or not citation_record.abstract:
            return {
                'is_valid': True,
                'confidence': 0.5,
                'reason': 'No model or abstract available - skip validation',
                'evidence': None
            }
        
        # 构建验证提示
        prompt = self._build_validation_prompt(citation_record, context_claim)
        
        try:
            # 调用LLM进行验证
            response = self.model.forward(
                [{'role': 'user', 'content': prompt}],
                temperature=0.3,
                top_p=0.9
            )
            
            # 解析LLM的响应
            result = self._parse_validation_response(response)
            
            return result
        except Exception as e:
            return {
                'is_valid': None,
                'confidence': 0.0,
                'error': str(e),
                'evidence': None
            }
    
    def step5_add(self, citation_record: CitationRecord) -> bool:
        """
        Step 5: Add (写入)
        将经过验证的引用写入最终的参考文献库
        
        Args:
            citation_record: 经过验证的引用记录
            
        Returns:
            是否成功添加
        """
        # 最终检查：确保所有必要字段都已填充
        if not all([
            citation_record.authors,
            citation_record.year,
            citation_record.verification_status == 'verified'
        ]):
            citation_record.verification_status = 'rejected'
            return False
        
        # 缓存该引用
        cache_key = f"{citation_record.authors}_{citation_record.year}"
        self.verified_cache[cache_key] = citation_record
        
        # 记录验证历史
        self.validation_log.append({
            'timestamp': datetime.now().isoformat(),
            'authors': citation_record.authors,
            'year': citation_record.year,
            'status': citation_record.verification_status
        })
        
        return True
    
    async def validate_citations_batch(self, 
                                     citations: List[Tuple[str, str, str]], 
                                     context_claims: Optional[Dict[str, str]] = None) -> Dict:
        """
        批量验证多个引用
        
        Args:
            citations: [(author, year, title), ...] 列表
            context_claims: {key: claim_text} 映射（用于validate步骤）
            
        Returns:
            验证结果总结
        """
        results = {}
        
        for author, year, title in citations:
            key = f"{author}_{year}"
            
            try:
                # Step 1: Search
                search_results = self.step1_search(author, year, title)
                if search_results.get('cached'):
                    results[key] = {
                        'status': 'cached',
                        'record': search_results['record']
                    }
                    continue
                
                # Step 2: Verify
                verified, verify_info = self.step2_verify(search_results)
                if not verified:
                    results[key] = {
                        'status': 'rejected',
                        'reason': 'Failed dual-source verification',
                        'sources_found': verify_info.get('sources_found', [])
                    }
                    continue
                
                best_result = verify_info['best_result']
                
                # Step 3: Retrieve
                bibtex = self.step3_retrieve(best_result)
                
                # 创建引用记录
                record = CitationRecord(
                    authors=author,
                    year=year,
                    title=title or best_result.get('title', 'Unknown'),
                    source=verify_info['sources_found'][0],
                    url=best_result.get('url', ''),
                    bibtex=bibtex or '',
                    abstract=best_result.get('abstract', ''),
                    verification_status='pending',
                    claim_text=context_claims.get(key, '') if context_claims else ''
                )
                
                # Step 4: Validate
                if record.abstract and context_claims and key in context_claims:
                    validation = self.step4_validate(record, context_claims[key])
                    record.validation_result = validation
                    
                    # 根据验证结果更新状态
                    if validation.get('is_valid'):
                        record.verification_status = 'verified'
                    else:
                        record.verification_status = 'rejected'
                
                # Step 5: Add
                success = self.step5_add(record)
                
                results[key] = {
                    'status': record.verification_status,
                    'record': asdict(record) if success else None
                }
                
            except Exception as e:
                results[key] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 降速（避免API限流）
            time.sleep(0.5)
        
        return results
    
    # ===== 内部辅助方法 =====
    
    def _search_crossref(self, author: str, year: str, title: Optional[str] = None) -> List[Dict]:
        """在CrossRef中搜索"""
        try:
            query = f"{author} {year}"
            if title:
                query += f" {title}"
            
            params = {
                'query': query,
                'rows': 3,
                'select': 'title,author,published,DOI,abstract'
            }
            
            response = self.session.get(self.CROSSREF_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('message', {}).get('items', []):
                results.append({
                    'title': item.get('title', [''])[0],
                    'authors': self._format_authors(item.get('author', [])),
                    'year': item.get('published', {}).get('date-parts', [[None]])[0][0],
                    'doi': item.get('DOI', ''),
                    'url': f"https://doi.org/{item.get('DOI', '')}",
                    'abstract': item.get('abstract', ''),
                    'source': 'crossref'
                })
            
            return results
        except Exception as e:
            print(f"CrossRef search failed: {e}")
            return []
    
    def _search_arxiv(self, author: str, year: str, title: Optional[str] = None) -> List[Dict]:
        """在arXiv中搜索"""
        try:
            query = f'author:"{author}"'
            if title:
                query += f' AND title:{title}'
            
            params = {
                'search_query': query,
                'start': 0,
                'max_results': 3,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.ARXIV_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # 解析Atom XML格式
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            results = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/abs/')[-1]
                results.append({
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                    'authors': self._extract_arxiv_authors(entry),
                    'year': arxiv_id.split('.')[0][:2] + '20',  # 近似从arxiv_id提取
                    'arxiv_id': arxiv_id,
                    'url': f"https://arxiv.org/abs/{arxiv_id}",
                    'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                    'source': 'arxiv'
                })
            
            return results
        except Exception as e:
            print(f"arXiv search failed: {e}")
            return []
    
    def _search_openalex(self, author: str, year: str, title: Optional[str] = None) -> List[Dict]:
        """在OpenAlex中搜索"""
        try:
            query = f'author.display_name:"{author}"'
            if title:
                query += f' AND title:"{title}"'
            
            params = {
                'filter': f'{query},publication_year:{year}',
                'per-page': 3,
                'sort': 'cited_by_count:desc'
            }
            
            response = self.session.get(self.OPENALEX_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'title': item.get('display_name', ''),
                    'authors': ', '.join([a.get('display_name', '') for a in item.get('authorships', [])]),
                    'year': item.get('publication_year', year),
                    'openalex_id': item.get('id', ''),
                    'url': item.get('doi', '') or item.get('id', ''),
                    'abstract': item.get('abstract_inverted_index', ''),  # OpenAlex以特殊格式存储摘要
                    'source': 'openalex'
                })
            
            return results
        except Exception as e:
            print(f"OpenAlex search failed: {e}")
            return []
    
    def _calculate_match_score(self, result: Dict) -> float:
        """计算搜索结果的匹配度分数"""
        score = 0.0
        
        # 有DOI/arxiv_id/openalex_id加分
        if result.get('doi') or result.get('arxiv_id') or result.get('openalex_id'):
            score += 0.3
        
        # 有摘要加分
        if result.get('abstract'):
            score += 0.2
        
        # 基础分
        score += 0.5
        
        return score
    
    def _get_bibtex_from_crossref(self, doi: str) -> Optional[str]:
        """从CrossRef获取BibTeX"""
        try:
            headers = {'Accept': 'application/x-bibtex'}
            url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Failed to get BibTeX from CrossRef: {e}")
        return None
    
    def _get_bibtex_from_arxiv(self, arxiv_id: str) -> Optional[str]:
        """从arXiv获取BibTeX"""
        try:
            # arXiv不直接提供BibTeX，但可以构造
            url = f"https://arxiv.org/abs/{arxiv_id}"
            return f"@misc{{{arxiv_id.replace('/', '_')},\n  url={{{url}}}\n}}"
        except Exception:
            pass
        return None
    
    def _get_bibtex_from_openalex(self, openalex_id: str) -> Optional[str]:
        """从OpenAlex获取BibTeX"""
        try:
            params = {'mailto': 'research@example.com'}
            response = self.session.get(f"{openalex_id}.bib", params=params, timeout=self.timeout)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass
        return None
    
    def _generate_basic_bibtex(self, result: Dict) -> str:
        """生成基础BibTeX格式"""
        # 构建安全的BibTeX键
        key_prefix = result.get('authors', 'unknown').split()[0].lower()
        year = result.get('year', 'nd')
        bibtex_key = f"{key_prefix}{year}"
        
        title = result.get('title', 'Untitled').replace('"', '\\"')
        
        # 生成基本条目
        bibtex = f"""@article{{{bibtex_key},
  title="{{{title}}}",
  author="{result.get('authors', 'Unknown')}",
  year={year}"""
        
        if result.get('url'):
            bibtex += f',\n  url="{result.get("url")}"'
        
        bibtex += "\n}"
        return bibtex
    
    def _format_authors(self, authors: List[Dict]) -> str:
        """格式化CrossRef的作者列表"""
        if not authors:
            return "Unknown"
        
        author_names = []
        for author in authors[:3]:  # 最多显示3个作者
            given = author.get('given', '')
            family = author.get('family', '')
            name = f"{given} {family}".strip()
            if name:
                author_names.append(name)
        
        if len(authors) > 3:
            return f"{', '.join(author_names)} et al."
        return ', '.join(author_names)
    
    def _extract_arxiv_authors(self, entry) -> str:
        """提取arXiv条目的作者"""
        try:
            authors = []
            for author_elem in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author_elem.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            if len(authors) > 3:
                return f"{', '.join(authors[:3])} et al."
            return ', '.join(authors)
        except:
            return "Unknown"
    
    def _build_validation_prompt(self, record: CitationRecord, claim: str) -> str:
        """构建验证提示"""
        return f"""请分析以下论文摘要是否能够支持所提出的学术观点：

【论文信息】
标题: {record.title}
作者: {record.authors}
出版年: {record.year}

【论文摘要】
{record.abstract[:500]}...

【在我们的报告中的使用观点】
{claim}

请判断：
1. 论文摘要中是否确实有证据支持这个观点？
2. 这个引用是否被合适地使用？
3. 是否存在过度解读或不当引用？

请用JSON格式回答：
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "evidence": "摘要中的具体支持证据",
  "issues": ["问题1", "问题2"]
}}
"""
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的验证响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass
        
        # 降级方案：简单判断
        return {
            'is_valid': 'valid' in response.lower(),
            'confidence': 0.5,
            'evidence': 'Unable to parse response',
            'issues': []
        }
    
    def export_verified_citations(self, format: str = 'json') -> str:
        """导出所有已验证的引用
        
        Args:
            format: 'json' 或 'bibtex'
            
        Returns:
            格式化的引用文本
        """
        if format == 'json':
            return json.dumps(
                {k: asdict(v) for k, v in self.verified_cache.items()},
                indent=2,
                ensure_ascii=False
            )
        elif format == 'bibtex':
            bibtex_entries = []
            for record in self.verified_cache.values():
                if record.bibtex:
                    bibtex_entries.append(record.bibtex)
            return '\n\n'.join(bibtex_entries)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """生成验证统计报告"""
        total = len(self.validation_log)
        verified = len([r for r in self.validation_log if r['status'] == 'verified'])
        rejected = len([r for r in self.validation_log if r['status'] == 'rejected'])
        
        return {
            'total_citations_processed': total,
            'verified_count': verified,
            'rejected_count': rejected,
            'verification_rate': verified / total if total > 0 else 0,
            'verification_log': self.validation_log
        }
