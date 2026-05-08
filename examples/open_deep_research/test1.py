import requests
import json
import time
import urllib.parse

# 你在 arXiv 中用的检索词组，对应到 OpenAlex 的 title.search
phrases = [
    '"3D heterogeneous integration"',
    '"three-dimensional heterogeneous integration"',
    '"monolithic 3D integration"',
    '"monolithic three-dimensional integration"',
    '"3D stacked integration"',
    '"three-dimensional stacked integration"',
    '"heterogeneous 3D stacking"',
    '"heterogeneous three-dimensional stacking"',
    '"hybrid bonding integration"',
    '"2.5D heterogeneous integration"',
    '"2.5-dimensional heterogeneous integration"'
]

# 构建 filter 表达式：title.search:短语1|短语2|...
search_terms = "|".join(phrases)

# 基础 URL
base_url = "https://api.openalex.org/works"

# 参数配置
params = {
    "filter": f"title.search:{search_terms},from_publication_date:2021-01-01,to_publication_date:2026-12-31",
    "sort": "publication_date:desc",
    "per_page": 200,
    "page": 1
}

print(f"Filter 参数: {params['filter']}\n")

def fetch_all_results(base_url, params, max_pages=10):
    """分页获取所有结果"""
    all_results = []
    
    for page in range(1, max_pages + 1):
        params["page"] = page
        
        print(f"⏳ 正在获取第 {page} 页...", end=" ")
        
        try:
            response = requests.get(base_url, params=params, headers={
                "User-Agent": "mailto:your_email@example.com"
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                meta = data.get("meta", {})
                total_count = meta.get("count", 0)
                
                all_results.extend(results)
                print(f"✓ 获取 {len(results)} 条，累计 {len(all_results)}/{total_count}")
                
                # 检查是否还有下一页
                if len(all_results) >= total_count or len(results) < params["per_page"]:
                    break
                
                time.sleep(0.1)
                
            elif response.status_code == 429:
                print("⚠ 被限流，等待 10 秒...")
                time.sleep(10)
                continue
            else:
                print(f"❌ 错误 {response.status_code}: {response.text[:100]}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            time.sleep(5)
            continue
    
    return all_results

# 获取数据
print("开始获取数据...\n")
all_works = fetch_all_results(base_url, params)

# 解析并整理数据
papers = []
for work in all_works:
    # 解析摘要
    abstract = ""
    if work.get("abstract_inverted_index") and isinstance(work["abstract_inverted_index"], dict):
        # 重建摘要
        inverted = work["abstract_inverted_index"]
        word_positions = []
        for word, positions in inverted.items():
            for pos in positions:
                word_positions.append((pos, word))
        # 按位置排序
        word_positions.sort(key=lambda x: x[0])
        abstract = " ".join([w for _, w in word_positions])
    
    # 解析作者
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)
    
    # 解析 DOI
    doi = work.get("doi", "")
    if doi:
        doi = doi.replace("https://doi.org/", "")
    
    paper = {
        "id": work.get("id", ""),
        "doi": doi,
        "title": work.get("title", ""),
        "publication_date": work.get("publication_date", ""),
        "authors": authors,
        "type": work.get("type", ""),
        "primary_topic": work.get("primary_topic", {}).get("display_name", "") if work.get("primary_topic") else "",
        "topics": [t.get("display_name", "") for t in work.get("topics", [])],
        "abstract": abstract,
        "open_access": work.get("open_access", {}).get("is_oa", False),
        "cited_by_count": work.get("cited_by_count", 0)
    }
    papers.append(paper)

# 构建最终 JSON
result = {
    "source": "OpenAlex",
    "search_type": "title.search",
    "date_range": "2021-01-01 to 2026-12-31",
    "total_results": len(papers),
    "papers": papers
}

# 保存文件
output_file = 'openalex_results_2021_2026.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完成！共获取 {len(papers)} 篇论文")
print(f"📁 结果已保存到: {output_file}")

# 打印简略摘要
print("\n" + "="*70)
print("📋 检索结果摘要：")
print("="*70)
for i, paper in enumerate(papers[:20], 1):
    authors_str = ", ".join(paper['authors'][:3])
    if len(paper['authors']) > 3:
        authors_str += " et al."
    print(f"\n{i}. {paper['title']}")
    print(f"   日期: {paper['publication_date']} | 类型: {paper['type']}")
    print(f"   作者: {authors_str}")
    print(f"   主题: {paper['primary_topic']}")
    if paper['doi']:
        print(f"   DOI: {paper['doi']}")
    print(f"   开放获取: {'是' if paper['open_access'] else '否'}")

if len(papers) > 20:
    print(f"\n... (还有 {len(papers) - 20} 篇未显示，完整结果见 JSON 文件)")