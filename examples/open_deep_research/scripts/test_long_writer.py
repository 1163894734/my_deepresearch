import os
from smolagents import OpenAIModel
from long_writer_tool import LongWriterTool # 确保路径正确

def test_long_writer():
    # 1. 初始化模型 (建议用你现有的配置)
    # 这里为了演示，假设你已经设置好了环境变量
    try:
        model = OpenAIModel(
            model_id="Qwen3-235B-A22B-Instruct-2507", # 替换为你实际可用的模型ID
            api_base="https://llmapi.paratera.com/v1",
            api_key=os.environ.get("DYM_API_KEY")
        )
    except Exception as e:
        print(f"模型初始化失败，请检查环境变量: {e}")
        return

    # 2. 初始化工具
    # 设置一个较小的 max_context_tokens 以便更容易触发“滑动窗口”逻辑
    lw_tool = LongWriterTool(model)

    # 3. 构造模拟参考资料 (Context)
    # 模拟包含一些论文信息的长文本
    mock_context = """
    文献1: Vaswani et al. (2017) "Attention Is All You Need". 介绍了Transformer架构。
    文献2: Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers".
    文献3: Brown et al. (2020) "Language Models are Few-Shot Learners". 介绍了GPT-3。
    文献4: Radford et al. (2019) "Language Models are Unsupervised Multitask Learners". 介绍了GPT-2。
    """

    # 4. 设置测试指令
    test_instruction = "写一篇关于NLP历史的短文，要求分为三个段落：早期模型、Transformer时代、大模型时代。最后列出参考文献。"

    print("=== 开始 LongWriterTool 压力测试 ===")
    try:
        # 执行工具
        result = lw_tool.forward(instruction=test_instruction, context=mock_context)
        
        print("\n=== 测试执行完成 ===")
        print(f"生成的总长度: {len(result)} 字符")
        
        # 5. 验证点检查
        print("\n--- 验证点检查 ---")
        
        # 检查是否包含参考文献
        if "参考文献" in result or "References" in result:
            print("[✅ PASS] 结尾包含参考文献部分")
        else:
            print("[❌ FAIL] 未发现参考文献部分")

        # 检查每一段后面是否干净（没有夹带参考文献）
        # 统计“参考文献”出现的次数，正常应该只在末尾出现1次
        ref_count = result.count("参考文献") + result.count("References")
        if ref_count == 1:
            print("[✅ PASS] 参考文献仅在末尾出现一次，未在段落间夹带")
        else:
            print(f"[⚠️ WARNING] 参考文献出现了 {ref_count} 次，请检查是否有残留")

        # 检查行内引用
        if "(" in result and "20" in result:
            print("[✅ PASS] 发现疑似 (Author, Year) 格式的行内引用")

        # 保存结果到本地查看
        with open("test_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n完整输出已保存至: test_output.txt")

    except Exception as e:
        print(f"[❌ ERROR] 运行中崩溃: {e}")

if __name__ == "__main__":
    test_long_writer()