"""极简OpenAlex连通性测试"""
import requests

# 改为你的邮箱
headers = {"User-Agent": "mailto:your@email.com"}

urls = [
    "https://api.openalex.org/works/W2741809807",
    "https://api.openalex.org/works?search=deep+learning&per_page=3",
]

for url in urls:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"✓ [{r.status_code}] {url[:60]}...")
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")