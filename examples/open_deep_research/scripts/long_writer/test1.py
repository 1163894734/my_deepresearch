import requests
# 测试调用一下无需 API Key 的公共接口
response = requests.get("https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b?fields=title,authors")
print(response.json())