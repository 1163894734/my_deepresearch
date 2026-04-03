import json
import os
import time

# 导入 smolagents 提供的真实大模型调用接口
# 这里以 LiteLLMModel 为例，你可以无缝切换为 HfApiModel 或 OpenAIServerModel
from smolagents import LiteLLMModel 

# 从你的主业务文件中导入 Agent 及其依赖
from scripts.identification_agent import (
    IdentificationAgent,
    BGEM3Embedder,
    UmapHdbscanClusterer,
)

def get_highly_realistic_docs() -> list[dict]:
    """
    提供高仿真的业务数据集。
    涵盖领域："AI Agent"
    包含两个潜在的细分流派：
    1. 多智能体协作 (Multi-Agent Collaboration) - 偏向软件开发、角色扮演
    2. 工具调用与交互 (Tool Use / API Calling) - 偏向外部系统集成
    """
    return [
        # ------ 流派 A: 多智能体协作 ------
        {
            "doc_id": "paper_2023_01",
            "timestamp": "2023-08-15",
            "title": "ChatDev: Communicative Agents for Software Development",
            "abstract": "The cognitive capabilities of large language models (LLMs) have advanced significantly. In this paper, we propose ChatDev, a virtual chat-powered software development company that operates through multiple agents holding different roles (e.g., programmer, reviewer, tester). Through collaborative dialogue, they complete the software development lifecycle. Our experiments demonstrate high efficiency and bug reduction.",
            "text": "multi-agent collaboration communicative agents software engineering LLM roles dialogue.",
        },
        {
            "doc_id": "paper_2024_02",
            "timestamp": "2024-02-10",
            "title": "AgentVerse: Facilitating Multi-Agent Environments",
            "abstract": "We introduce AgentVerse, a versatile framework that enables researchers to easily build custom multi-agent environments. It provides standardized interfaces for agent communication, role assignment, and task evaluation. We demonstrate its application in collaborative writing, debate, and game playing, showing that multi-agent systems consistently outperform single-agent baselines in complex reasoning tasks.",
            "text": "agentverse framework multi-agent environments communication protocol reasoning.",
        },
        {
            "doc_id": "paper_2024_03",
            "timestamp": "2024-05-22",
            "title": "Auto-Reviewer: Dual-Agent Conflict Resolution in Code Generation",
            "abstract": "To mitigate hallucinations in LLM code generation, we propose a dual-agent system consisting of a Generator Agent and a Critic Agent. They engage in a zero-sum debate to find logical flaws. This adversarial multi-agent workflow increases the pass@1 rate on HumanEval by 14.5% compared to monolithic models.",
            "text": "adversarial agents dual-agent code generation debate conflict resolution.",
        },
        {
            "doc_id": "paper_2025_04",
            "timestamp": "2025-01-11",
            "title": "Massive Multi-Agent Swarms for Automated Data Annotation",
            "abstract": "Data annotation is expensive. We deploy swarms of thousands of micro-agents, each assigned a micro-task and a specific persona, to collaboratively label and verify massive datasets. We introduce a novel consensus algorithm for agent voting, reducing annotation costs by 90% while maintaining human-level accuracy.",
            "text": "agent swarms data annotation consensus algorithm micro-agents.",
        },

        # ------ 流派 B: 工具调用与外部交互 ------
        {
            "doc_id": "paper_2023_05",
            "timestamp": "2023-05-24",
            "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
            "abstract": "Language models struggle with tasks requiring external knowledge or precise math. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, and how to parse the results. It integrates search engines, calculators, and translation systems natively into its generation process via self-supervised API tokens.",
            "text": "toolformer API calling external tools search engine calculator integration.",
        },
        {
            "doc_id": "paper_2023_06",
            "timestamp": "2023-11-05",
            "title": "Gorilla: Large Language Model Connected with Massive APIs",
            "abstract": "We present Gorilla, an LLM specifically finetuned to write API calls. By combining document retrieval with instruction tuning, Gorilla surpasses GPT-4 in generating accurate API requests for AWS, GCP, and Hugging Face services, significantly reducing hallucinated arguments.",
            "text": "gorilla API calls instruction tuning massive APIs documentation retrieval.",
        },
        {
            "doc_id": "paper_2024_07",
            "timestamp": "2024-06-18",
            "title": "Executable Agentic Workflows for Database Management",
            "abstract": "This study explores the use of autonomous agents for SQL database administration. The agent is equipped with tools to execute queries, read schemas, and rollback transactions. By enabling the agent to observe the execution environment and adjust its SQL logic iteratively, we achieve zero-shot database optimization.",
            "text": "database administration SQL agent execution environment tool use.",
        },
        {
            "doc_id": "paper_2025_08",
            "timestamp": "2025-03-02",
            "title": "Self-Correcting Agents via Sandbox Execution Feedback",
            "abstract": "We propose a novel framework where an LLM agent writes Python code and executes it within a secure Docker sandbox. The agent captures the runtime errors or standard output as feedback, and iteratively corrects its code. This tool-use feedback loop allows the agent to solve highly complex mathematical modeling tasks.",
            "text": "sandbox execution self-correction Python execution runtime feedback loop.",
        }
    ]

def main():
    print("🚀 [系统启动] 正在准备真实场景测试...")

    # ==========================================
    # 1. 配置真实的大模型 (以 OpenAI 兼容接口为例)
    # ==========================================
    # 请确保你的环境变量中设置了对应的 API KEY
    # 例如：export OPENAI_API_KEY="sk-xxxx"
    # 或者如果你用的是 DeepSeek/通义千问，可以修改 api_base
    try:
        model = LiteLLMModel(
            model_id="gpt-4o", # 替换为你实际使用的模型，如 "deepseek/deepseek-chat"
            temperature=0.2,   # 保持较低温度以保证 JSON 输出的稳定性
        )
    except Exception as e:
        print(f"❌ 模型初始化失败，请检查环境变量或依赖: {e}")
        return

    # ==========================================
    # 2. 组装具备真实数学计算能力的 Agent
    # ==========================================
    # 强制启用 BGE-M3 和 UMAP+HDBSCAN 进行向量化聚类
    print("⚙️ [组件加载] 正在加载 BGE-M3 向量模型与 HDBSCAN 聚类器...")
    print("   (初次运行可能需要下载模型权重，请耐心等待)")
    try:
        embedder = BGEM3Embedder(model_name="BAAI/bge-m3", batch_size=8)
        clusterer = UmapHdbscanClusterer(umap_dim=5, min_cluster_size=2) # 数据量少，调低 min_cluster_size
    except ImportError as e:
        print(f"❌ 缺少核心科学计算库，请先执行: pip install sentence-transformers umap-learn hdbscan scikit-learn")
        print(f"详细错误: {e}")
        return

    # 领域判别器：确保处理的是 AI Agent 相关的文章
    classifier = KeywordDomainClassifier(
        domain_keywords={"ai_agent": ["agent", "llm", "tool", "api", "multi-agent", "swarm"]}
    )

    agent = IdentificationAgent(
        model=model,
        classifier=classifier,
        embedder=embedder,
        clusterer=clusterer,
        # 内涵生成器和五专家在 Agent 内部会自动使用传入的 model，无需手动挂载
    )

    # ==========================================
    # 3. 构造请求 Payload 并执行测试
    # ==========================================
    payload = {
        "domain": "ai_agent",
        "documents": get_highly_realistic_docs(),
        "min_domain_probability": 0.3,
        "top_k_per_cluster": 3,
        "minimum_votes_to_pass": 3, # 必须至少3个专家同意
    }

    print(f"\n🧠 [开始推理] 向 Agent 输入 {len(payload['documents'])} 篇仿真前沿文献...")
    start_time = time.time()
    
    try:
        # 直接调用 run()，这会触发完整的 Pipeline
        result_json_str = agent.run(json.dumps(payload, ensure_ascii=False))
        result_obj = json.loads(result_json_str)
    except Exception as e:
        print(f"\n❌ [执行异常] Pipeline 崩溃: {e}")
        return

    cost_time = time.time() - start_time

    # ==========================================
    # 4. 打印高可读性的真实业务报告
    # ==========================================
    print(f"\n✅ [测试完成] 耗时: {cost_time:.1f} 秒")
    print("\n" + "="*50)
    print("📊 最终前沿技术识别报告 (Final Report)")
    print("="*50)
    print(f"🔹 目标领域: {result_obj.get('domain')}")
    print(f"🔹 输入文献: {result_obj.get('total_input_docs')} 篇")
    print(f"🔹 有效保留: {result_obj.get('kept_docs')} 篇 | 噪音剔除: {result_obj.get('removed_docs')} 篇")
    
    clusters = result_obj.get('clusters', [])
    print(f"🔹 成功突围的技术簇: {len(clusters)} 个\n")

    for i, cluster in enumerate(clusters, 1):
        print(f"🚀 【技术簇 {i}】: {cluster.get('technology_term')}")
        print(f"   🔸 包含文献数: {cluster.get('size')} 篇")
        print(f"   🔸 解决痛点: {cluster.get('technology_problem')}")
        print(f"   🔸 核心手段: {cluster.get('technology_method')}")
        print(f"   🔸 商业指向: {cluster.get('application_direction')}")
        
        trend = cluster.get('trend', {})
        print(f"   📈 演化特征: CAGR={trend.get('cagr', 0):.2%} | 是否突现: {'是' if trend.get('burst_detected') else '否'}")
        print(f"      (突现理由: {trend.get('burst_reason')})")

        print("   🧑‍⚖️ 专家评审委员会决议:")
        votes = cluster.get('votes', [])
        approve_count = sum(1 for v in votes if v.get('approve'))
        print(f"      [得票数: {approve_count}/{len(votes)}] -> {'🏆 准予立项' if cluster.get('approved') else '⛔ 予以否决'}")
        
        for vote in votes:
            icon = "✅" if vote.get('approve') else "❌"
            print(f"      - {icon} {vote.get('expert_name')}: {vote.get('reason')}")
        print("-" * 40)

if __name__ == "__main__":
    main()