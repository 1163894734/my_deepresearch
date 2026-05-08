import json
import os
import requests
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def load_papers(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('papers', [])

def sanitize_filename(title):
    title = re.sub(r'<[^>]+>', '', title)
    title = re.sub(r'[<>:"/\\|?*]', '_', title)
    title = re.sub(r'\s+', ' ', title).strip()
    if len(title) > 150:
        title = title[:150]
    return title

def get_arxiv_id(paper):
    """提取 arXiv ID"""
    doi = paper.get('doi', '')
    # 匹配 10.48550/arxiv.xxxx.xxxxx 格式
    m = re.search(r'10\.48550/arxiv\.(.+)', doi)
    if m:
        return m.group(1)
    # 也检查 id 字段
    paper_id = paper.get('id', '')
    if 'arxiv' in paper_id.lower():
        parts = paper_id.split('/')
        if len(parts) > 1:
            return parts[-1]
    return None

def get_pdf_url(paper):
    """根据 DOI 和论文类型构造 PDF URL"""
    doi = paper.get('doi', '')
    paper_type = paper.get('type', '')
    
    # ===== arXiv 预印本优先 =====
    arxiv_id = get_arxiv_id(paper)
    if arxiv_id:
        # 去掉版本号
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        url = f"https://arxiv.org/pdf/{clean_id}"
        return url, "arXiv"
    
    if not doi:
        return None, "无DOI"
    
    # ===== 根据 DOI 前缀匹配出版社 =====
    
    # Nature 系列
    if any(k in doi for k in ['10.1038/', '10.1038/s41586', '10.1038/s41565', 
                                '10.1038/s41467', '10.1038/s41928', '10.1038/s41699']):
        suffix = doi.split('/')[-1]
        return f"https://www.nature.com/articles/{suffix}.pdf", "Nature"
    
    # Wiley 系列
    if any(k in doi for k in ['10.1002/', '10.1002/adfm', '10.1002/adma', 
                                '10.1002/admt', '10.1002/inf2']):
        return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true", "Wiley"
    
    # ACS
    if '10.1021/' in doi:
        return f"https://pubs.acs.org/doi/pdf/{doi}?download=true", "ACS"
    
    # IEEE (用 ieeexplore)
    if '10.1109/' in doi:
        return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={doi.split('/')[-1]}", "IEEE"
    
    # Elsevier
    if '10.1016/' in doi:
        pii = doi.replace('10.1016/', '')
        return f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?md5=xxx&pid=1-s2.0-{pii}-main.pdf", "Elsevier"
    
    # MDPI
    if '10.3390/' in doi:
        parts = doi.replace('10.3390/', '').split('/')
        return f"https://www.mdpi.com/{parts[0]}/{parts[1]}/pdf", "MDPI"
    
    # IOP
    if '10.1088/' in doi:
        return f"https://iopscience.iop.org/article/{doi}/pdf", "IOP"
    
    # OUP (Oxford)
    if '10.1093/' in doi:
        return f"https://academic.oup.com/nsr/advance-article-pdf/doi/{doi}", "OUP"
    
    # SSRN 预印本
    if '10.2139/' in doi:
        ssrn_id = doi.split('/')[-1]
        return f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}", "SSRN"
    
    # 通用: 通过 doi.org 重定向
    return f"https://doi.org/{doi}", "DOI重定向"

def download_with_retry(url, filepath, source, max_retries=3):
    """下载 PDF，带重试和智能 User-Agent"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # 根据不同来源加 Referer
    if "arxiv.org" in url:
        headers["Accept"] = "application/pdf"
    elif "nature.com" in url:
        headers["Referer"] = "https://www.nature.com/"
    elif "wiley.com" in url:
        headers["Referer"] = "https://onlinelibrary.wiley.com/"
    elif "acs.org" in url:
        headers["Referer"] = "https://pubs.acs.org/"
    elif "ieee.org" in url:
        headers["Referer"] = "https://ieeexplore.ieee.org/"
    elif "mdpi.com" in url:
        headers["Referer"] = "https://www.mdpi.com/"
    elif "iopscience" in url:
        headers["Referer"] = "https://iopscience.iop.org/"
    else:
        headers["Referer"] = "https://scholar.google.com/"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(3 * attempt)  # 重试前等待
            
            response = requests.get(url, headers=headers, timeout=60, 
                                    allow_redirects=True, stream=True)
            
            # arXiv 404 处理：尝试带版本号的 URL
            if response.status_code == 404 and "arxiv.org" in url and 'v' not in url.split('/')[-1]:
                # 可能需要对版本号，但这里先跳过
                safe_print(f"  ⚠ arXiv 404 (可能版本号问题): {url}")
                return False
            
            if response.status_code == 200:
                content = response.content
                
                # 检查是否真的是 PDF
                if content[:4] == b'%PDF':
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    safe_print(f"  ✅ [{source}] 下载成功 ({len(content)//1024} KB)")
                    return True
                elif b'<html' in content[:200].lower() or b'<!doctype' in content[:200].lower():
                    safe_print(f"  ❌ [{source}] 返回 HTML 而非 PDF (可能需要登录)")
                    return False
                else:
                    # 未知格式，但也保存
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    safe_print(f"  ⚠ [{source}] 内容格式未知，已保存 ({len(content)//1024} KB)")
                    return True
            elif response.status_code == 403:
                safe_print(f"  🔒 [{source}] 403 Forbidden (尝试 {attempt+1}/{max_retries})")
                # 换一个 User-Agent 再试
                if attempt == 1:
                    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                continue
            else:
                safe_print(f"  ❌ [{source}] HTTP {response.status_code}")
                return False
                
        except Exception as e:
            safe_print(f"  ❌ [{source}] 异常: {e}")
            if attempt < max_retries - 1:
                continue
            return False
    
    return False

def process_paper(paper, output_dir, index, total):
    """处理单篇论文"""
    title = paper.get('title', 'Unknown Title')[:80]
    doi = paper.get('doi', '')
    arxiv_id = get_arxiv_id(paper)
    
    safe_print(f"\n[{index}/{total}] {title}")
    safe_print(f"  DOI: {doi} | arXiv: {arxiv_id or '无'} | OA: {paper.get('open_access', False)}")
    
    # 文件名
    filename = sanitize_filename(paper.get('title', 'untitled')) + ".pdf"
    filepath = output_dir / filename
    
    # 跳过已存在的有效 PDF
    if filepath.exists() and filepath.stat().st_size > 500:
        with open(filepath, 'rb') as f:
            header = f.read(4)
        if header == b'%PDF':
            safe_print(f"  ⏭ 已存在")
            return paper, True
    
    # 获取下载链接
    pdf_url, source = get_pdf_url(paper)
    
    if not pdf_url:
        safe_print(f"  ⚠ 无下载链接")
        return paper, False
    
    # 下载
    success = download_with_retry(pdf_url, filepath, source)
    return paper, success

def main():
    json_path = "openalex_results_2021_2026.json"
    output_dir = Path("downloaded_papers")
    output_dir.mkdir(exist_ok=True)
    
    print("📂 加载论文数据...")
    papers = load_papers(json_path)
    print(f"📄 共 {len(papers)} 篇论文\n")
    
    max_workers = 3  # 降低并发，减少被封概率
    success_list = []
    failed_list = []
    
    print("="*60)
    print(f"⏬ 开始下载 (并发: {max_workers})...")
    print("="*60)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_paper, paper, output_dir, i+1, len(papers)): paper
            for i, paper in enumerate(papers)
        }
        
        for future in as_completed(futures):
            paper, success = future.result()
            if success:
                success_list.append(paper)
            else:
                failed_list.append(paper)
    
    # 统计
    print("\n" + "="*60)
    print(f"📊 下载完成: ✅ {len(success_list)} | ❌ {len(failed_list)}")
    print(f"📁 文件目录: {output_dir.absolute()}")
    
    # 失败列表
    if failed_list:
        print(f"\n❌ 失败的论文 ({len(failed_list)} 篇):")
        for p in failed_list[:20]:
            arxiv_id = get_arxiv_id(p)
            has_arxiv = "有arXiv" if arxiv_id else "无arXiv"
            print(f"  - [{has_arxiv}] {p.get('doi','无DOI')} | {p.get('title','')[:60]}")

if __name__ == "__main__":
    main()