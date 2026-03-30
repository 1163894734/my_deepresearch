from openai import OpenAI
import sys

# 初始化客户端，指向 LM Studio 本地服务器
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# 1. 初始化对话历史列表（可以在这里预设 System 角色）
messages = [
    {"role": "system", "content": "你是一个严谨且乐于助人的 AI 助手。"}
]

print("已连接本地大模型！(输入 'exit' 或 'quit' 退出对话)")
print("-" * 50)

# 2. 使用 while 循环实现持续的命令行交互
while True:
    # 获取用户输入
    user_input = input("\n你: ")
    
    # 退出命令检测
    if user_input.strip().lower() in ['exit', 'quit']:
        print("对话已结束。")
        break
        
    # 防止空输入触发无意义请求
    if not user_input.strip():
        continue

    # 3. 将当前用户的问题追加到历史记录中
    messages.append({"role": "user", "content": user_input})
    
    print("模型: ", end="")
    
    try:
        # 发送包含完整历史记录的 messages 列表
        stream = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=True
        )
        
        # 用于临时拼接模型当前的完整回复
        full_response = ""
        
        # 遍历数据流并实时打印
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                print(content, end="", flush=True)
                full_response += content # 将每一个字拼接到完整回复中
                
        print() # 当前轮次输出结束后换行
        
        # 4. 将模型的完整回复追加到历史记录中，形成完整的上下文闭环
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        print(f"\n[请求出错，请检查 LM Studio 服务是否开启] 错误信息: {e}")