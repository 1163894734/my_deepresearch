from openai import OpenAI

# 指向 LM Studio 本地服务器
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # 本地服务器不需要真实 API key
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "什么是自监督学习？"}
    ]
)

print(response.choices[0].message.content)