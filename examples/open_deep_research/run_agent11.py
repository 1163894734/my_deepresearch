import argparse
import asyncio
import os
import time

# 🔥 直接导入你已经写好的完美多智能体流水线
from scripts.multi_agent.run_pipeline import run_long_writer_pipeline

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Long Writer Pipeline")
    parser.add_argument("question", type=str, help="Research task or topic")
    args = parser.parse_args()

    # 创建专属的输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./outputs", f"report_{timestamp}")

    print(f"\n🚀 接收到任务: {args.question}")
    print(f"📁 准备输出到: {output_dir}\n")

    # 启动异步事件循环，直接调用底层模块
    asyncio.run(run_long_writer_pipeline(args.question, output_dir))

    print(f"\n🎉 流水线执行完毕，请前往 {output_dir} 验收成果！")

if __name__ == "__main__":
    main()