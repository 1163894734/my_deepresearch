import requests
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time

# 你的完整检索词
query_string = '(("3D heterogeneous integration" OR "three-dimensional heterogeneous integration") OR ("monolithic 3D integration" OR "monolithic three-dimensional integration") OR ("3D stacked integration" OR "three-dimensional stacked integration") OR ("heterogeneous 3D stacking" OR "heterogeneous three-dimensional stacking") OR ("hybrid bonding integration") OR ("2.5D heterogeneous integration" OR "2.5-dimensional heterogeneous integration"))'

# URL 编码
encoded_query = urllib.parse.quote(query_string)

# 拼接 API 请求 URL，增加时间限制参数
# submittedDate: 从2021-01-01到2026-12-31
base_url = "https://export.arxiv.org/api/query"
url = (
    f"{base_url}?search_query=all:{encoded_query}"
    f"&start=0&max_results=100"  # 每次最多返回100条，先设大一些看看有多少
    f"&sortBy=submittedDate&sortOrder=descending"  # 按提交时间降序排列
)

print(f"请求的URL: {url}\n")

def make_request(url, max_retries=5):
    """发送请求，遇到 429 自动等待重试"""
    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 10 * (attempt + 1)
            print(f"⚠️ 第 {attempt + 1} 次尝试被限流，等待 {wait_time} 秒...")
            time.sleep(wait_time)
        else:
            print("⏳ 首次请求前等待 5 秒，避免触发限流...")
            time.sleep(5)
        
        response = requests.get(url, headers={
            "User-Agent": "MyResearchBot/1.0 (your_email@example.com)"
        })
        
        print(f"第 {attempt + 1} 次尝试 - HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            continue
        else:
            print(f"❌ 未知错误 (HTTP {response.status_code}): {response.text[:200]}")
            return None
    
    print("❌ 重试耗尽，请求失败")
    return None

# 发送请求
response = make_request(url)

if response is None:
    exit()

if not response.text.strip():
    print("❌ 返回内容为空")
    exit()

print("✅ 请求成功，开始解析 XML...\n")

# 解析 XML
try:
    root = ET.fromstring(response.text)
except ET.ParseError as e:
    print(f"❌ XML 解析错误: {e}")
    print(f"实际返回内容前500字符:\n{response.text[:500]}")
    exit()

# 定义命名空间
namespaces = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
}

# 提取元数据
total_results = root.find('opensearch:totalResults', namespaces)
start_index = root.find('opensearch:startIndex', namespaces)
items_per_page = root.find('opensearch:itemsPerPage', namespaces)

print(f"📊 总结果数: {total_results.text if total_results is not None else '未知'}")
print(f"📍 起始索引: {start_index.text if start_index is not None else '未知'}")
print(f"📄 每页条数: {items_per_page.text if items_per_page is not None else '未知'}\n")

# 时间边界
DATE_FROM = "2021-01-01T00:00:00Z"
DATE_UNTIL = "2026-12-31T23:59:59Z"

# 解析每篇论文，并手动过滤时间
papers = []
filtered_count = 0
for entry in root.findall('atom:entry', namespaces):
    pub_date_str = entry.find('atom:published', namespaces).text.strip() if entry.find('atom:published', namespaces) is not None else ""
    
    # 时间过滤：只保留2021-2026年之间的论文
    if pub_date_str < DATE_FROM or pub_date_str > DATE_UNTIL:
        filtered_count += 1
        continue
    
    paper = {
        "id": entry.find('atom:id', namespaces).text.strip() if entry.find('atom:id', namespaces) is not None else "",
        "title": entry.find('atom:title', namespaces).text.strip().replace('\n', ' ') if entry.find('atom:title', namespaces) is not None else "",
        "summary": entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ') if entry.find('atom:summary', namespaces) is not None else "",
        "published": pub_date_str,
        "updated": entry.find('atom:updated', namespaces).text.strip() if entry.find('atom:updated', namespaces) is not None else "",
        "authors": [],
        "doi": "",
        "journal_ref": "",
        "primary_category": "",
        "categories": [],
        "links": []
    }
    
    # 作者
    for author in entry.findall('atom:author', namespaces):
        name = author.find('atom:name', namespaces)
        if name is not None:
            paper["authors"].append(name.text.strip())
    
    # DOI
    doi_elem = entry.find('arxiv:doi', namespaces)
    if doi_elem is not None:
        paper["doi"] = doi_elem.text.strip()
    
    # 期刊引用
    journal_elem = entry.find('arxiv:journal_ref', namespaces)
    if journal_elem is not None:
        paper["journal_ref"] = journal_elem.text.strip()
    
    # 主分类
    primary_cat = entry.find('arxiv:primary_category', namespaces)
    if primary_cat is not None:
        paper["primary_category"] = primary_cat.get('term', '')
    
    # 全部分类
    for cat in entry.findall('atom:category', namespaces):
        paper["categories"].append(cat.get('term', ''))
    
    # 链接
    for link in entry.findall('atom:link', namespaces):
        paper["links"].append({
            "href": link.get('href', ''),
            "rel": link.get('rel', ''),
            "title": link.get('title', '')
        })
    
    papers.append(paper)
    print(f"  📝 [{pub_date_str[:10]}] {paper['title'][:80]}...")

print(f"\n🔍 时间过滤统计：")
print(f"  - 原始结果数: {int(total_results.text) if total_results is not None else 0}")
print(f"  - 过滤掉（日期<{DATE_FROM[:4]}或>{DATE_UNTIL[:4]}）: {filtered_count} 篇")
print(f"  - 最终保留: {len(papers)} 篇")

# 构建 JSON 结果
result = {
    "query": query_string,
    "date_range": {
        "from": "2021-01-01",
        "until": "2026-12-31"
    },
    "total_results_filtered": len(papers),
    "start_index": int(start_index.text) if start_index is not None else 0,
    "items_per_page": int(items_per_page.text) if items_per_page is not None else 0,
    "papers": papers
}

# 保存到文件
output_file = 'arxiv_results_2021_2026.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完成！共获取 {len(papers)} 篇论文（2021-2026年）")
print(f"📁 结果已保存到: {output_file}")

# 控制台打印简略摘要
print("\n" + "="*60)
print("📋 检索结果摘要：")
print("="*60)
for i, paper in enumerate(papers, 1):
    authors = ", ".join(paper['authors'][:3])
    if len(paper['authors']) > 3:
        authors += " et al."
    print(f"\n{i}. {paper['title']}")
    print(f"   日期: {paper['published'][:10]} | 作者: {authors}")
    print(f"   分类: {paper['primary_category']}")
    if paper['doi']:
        print(f"   DOI: {paper['doi']}")