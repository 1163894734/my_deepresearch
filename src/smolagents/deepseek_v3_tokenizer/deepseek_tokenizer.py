# pip3 install transformers
# python3 deepseek_tokenizer.py
import transformers
import os

# chat_tokenizer_dir = "./"
chat_tokenizer_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = transformers.AutoTokenizer.from_pretrained( 
        chat_tokenizer_dir, trust_remote_code=True
        )

# # result = tokenizer.encode("Hello! my name is Deepseek.")
# # print(len(result))
# def print_tokens(str):
#     return len(tokenizer.encode(str))
# def print_msg_tokens(messages):
#     """简单拼接所有消息内容"""
#     result = []
#     for msg in messages:
#         # msg 是 gr.ChatMessage 对象
#         role = msg['role']  # "user" 或 "assistant"
#         content = msg['content']
#         result.append(f"{role}: {content}")
    
#     res = "\n".join(result)
#     return len(tokenizer.encode(res))
# # 打印./的绝对路径


# def print_tokens(messages) -> int:  # 暂时不指定 messages 的类型注解，避免依赖
#     """提取消息列表中的所有文本内容并计算 token 数"""
#     # 延迟导入：在函数内部导入，避免模块加载时的循环依赖
#     from smolagents.models import ChatMessage

#     # 提取所有消息中的文本内容
#     all_text = []
#     for msg in messages:
#         # 验证消息类型
#         if not isinstance(msg, ChatMessage):
#             continue  # 或抛出异常：raise ValueError("Invalid message type")
#         # 处理 content 中的文本
#         if isinstance(msg.content, list):
#             text_parts = [item["text"] for item in msg.content if item.get("type") == "text"]
#             all_text.extend(text_parts)
    
#     full_text = " ".join(all_text)
#     return len(tokenizer.encode(full_text))

def print_tokens(str) -> int:
    return len(tokenizer.encode(str))