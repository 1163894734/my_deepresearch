from smolagents.tools import Tool


context =  """
[DeepSeek-V3 Technical Report + 2024 + DeepSeek AI]：详细介绍了DeepSeek-V3的架构（MoE + MLA），展示了在极低成本下训练最强开源模型的方法，以及多头潜在注意力（MLA）在显存优化上的贡献。

[The Llama 3 Herd of Models + 2024 + Meta AI]：全面披露了Llama 3系列的训练细节，包括从8B到405B的扩展定律、数据清洗流程及后训练（Post-training）策略。

[Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context + 2024 + Google DeepMind]：重点介绍了混合专家（MoE）架构在处理超长上下文（10M+ tokens）和多模态理解方面的突破。

[Qwen2.5 Technical Report + 2024 + Alibaba Cloud]：展示了Qwen2.5在编码、数学和多语言能力上的显著提升，强调了合成数据在预训练中的作用。

[GPT-4 Technical Report + 2023 + OpenAI]：虽然未公开细节，但确立了多模态大模型的行业基准，强调了安全性测试和RLHF的重要性。

[Mistral Large + 2024 + Mistral AI]：展示了欧洲顶尖开源模型的实力，强调了在推理效率和多语言处理上的优化。

[Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone + 2024 + Microsoft]：证明了通过高质量的数据（"Textbooks Are All You Need"），小参数模型也能达到极高性能。

[Yi: Open Foundation Models by 01.AI + 2024 + 01.AI]：李开复团队的模型，强调了在中文语境和超长窗口下的优异表现。

[Nemotron-4 340B Technical Report + 2024 + NVIDIA]：专注于为合成数据生成而优化的模型，展示了如何用大模型训练小模型。

[Gemma: Open Models Based on Gemini Research and Technology + 2024 + Google]：Google开源的轻量级模型系列，技术同源于Gemini。

二、 架构创新与高效计算（Architecture & Efficiency）
Transformer的改进与替代架构。

[Mamba: Linear-Time Sequence Modeling with Selective State Spaces + 2023 + Gu, Dao et al.]：提出了SSM（状态空间模型）架构，实现了线性时间的推理速度，是Transformer的最强挑战者。

[Jamba: A Hybrid Transformer-Mamba Language Model + 2024 + AI21 Labs]：结合了Mamba的高效和Transformer的高质量，提出了混合架构的新方向。

[LoRA: Low-Rank Adaptation of Large Language Models + 2021/2022 + Hu et al.]：微调领域的基石，通过低秩分解实现极低参数量的适配，至今仍是主流。

[QLoRA: Efficient Finetuning of Quantized LLMs + 2023 + Dettmers et al.]：将量化与LoRA结合，使得在单张消费级显卡上微调33B/65B模型成为可能。

[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision + 2024 + Dao et al.]：进一步优化了Attention机制的计算速度，是大模型训练加速的核心技术。

[Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models + 2024 + Google DeepMind]：提出动态分配计算资源，不同token经过不同层数的处理，提高效率。

[Ring Attention with Blockwise Transformers for Near-Infinite Context + 2023 + Liu et al.]：解决了超长上下文的显存瓶颈，支持百万级token的训练。

[KVQuant: LLM Inference with Low-Precision Key-Value Cache + 2024 + Various]：针对KV Cache的量化技术，显著降低长文本推理的显存占用。

[BitNet: Scaling 1-bit Transformers for Large Language Models + 2023 + Microsoft]：探索了1-bit量化训练，为未来超低能耗大模型指明方向。

[Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits + 2024 + Microsoft]：BitNet的进阶版，证明了三元权重（-1, 0, 1）在保持性能的同时极大降低计算成本。

三、 对齐与训练策略（Alignment, RLHF & Training）
如何让模型听话、安全、符合人类价值观。

[Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO) + 2023 + Rafailov et al.]：革命性地去除了RLHF中的Reward Model训练步骤，直接在偏好数据上优化，成为当前主流对齐算法。

[Constitutional AI: Harmlessness from AI Feedback + 2022 + Anthropic]：提出了RLAIF（AI反馈强化学习），利用AI生成的反馈来微调AI，减少人类标注成本。

[KTO: Model Alignment as Prospect Theoretic Optimization + 2024 + Ethayarajh et al.]：无需成对偏好数据（A胜过B），仅需点赞/点踩数据即可进行对齐。

[Self-Rewarding Language Models + 2024 + Meta AI]：模型自己生成指令并自己评分，实现自我迭代进化。

[SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Learners + 2024 + Chen et al.]：通过自我博弈机制，在没有额外人工标注的情况下提升模型性能。

[NEFTune: Noisy Embeddings Improve Instruction Finetuning + 2023 + Jain et al.]：在Embedding层加入噪声，显著提升了指令微调的鲁棒性和对话质量。

[Orpo: Monolithic Preference Optimization without Reference Model + 2024 + Hong et al.]：无需参考模型（Reference Model）的偏好优化方法，节省显存并提高训练速度。

[SimPO: Simple Preference Optimization with a Reference-Free Reward + 2024 + Meng et al.]：比DPO更简单、更高效的离线偏好优化算法。

[Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision + 2023 + OpenAI]：Ilya Sutskever团队作品，探讨如何用弱模型监督强模型，是超级对齐（Superalignment）的核心。

[Training Language Models to Follow Instructions with Human Feedback (InstructGPT) + 2022 + OpenAI]：RLHF的开山之作，理解ChatGPT起源的必读文献。

四、 推理与思维链（Reasoning & Chain of Thought）
如何提升模型的逻辑推理和数学能力。

[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models + 2022 + Wei et al.]：CoT鼻祖，发现让模型“一步步思考”能剧烈提升推理能力。

[Self-Consistency Improves Chain of Thought Reasoning in Language Models + 2022 + Wang et al.]：通过采样多条推理路径并投票，显著提升CoT的准确率。

[Tree of Thoughts: Deliberate Problem Solving with Large Language Models + 2023 + Yao et al.]：引入树搜索算法（BFS/DFS），让模型在推理过程中进行探索和回溯。

[Let's Verify Step by Step + 2023 + OpenAI]：提出了过程监督（Process Supervision），即对推理的每一步进行打分，而非仅对结果打分。

[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking + 2024 + Zelikman et al.]：让模型在输出每个token前在内部进行隐式推理（类似内心独白）。

[CoT-Decoding: Confidence-Guided Decoding for Reasoning + 2024 + DeepMind]：解码阶段的优化，利用CoT的置信度来引导生成。

[Math-Shepherd: Verify and Reinforce Langauage Models Step-by-step without Human Annotations + 2023 + Lightman et al.]：专注于数学推理的过程奖励模型自动化构建。

[Everything of Thoughts (XOT): End-to-End Thought Retrieval + 2023 + Ding et al.]：结合检索和思维链，利用外部知识库辅助推理。

[Graph of Thoughts: Solving Elaborate Problems with Large Language Models + 2023 + Besta et al.]：将推理过程建模为图结构，允许任意的跳跃和聚合。

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning + 2025 + DeepSeek]：证明了仅通过强化学习（无SFT）可以激发极强的推理能力。

五、 RAG与长上下文（Retrieval-Augmented Generation & Long Context）
解决幻觉和知识时效性问题。

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks + 2020/2023 + Lewis et al.]：RAG的基础概念，虽然早但必读，后续有大量2023-2024的改进版。

[Lost in the Middle: How Language Models Use Long Contexts + 2023 + Liu et al.]：发现模型在长文本中容易忽略中间信息，对长上下文RAG设计有重要指导意义。

[GraphRAG: Unlocking LLM Discovery on Narrative Private Data + 2024 + Microsoft]：利用知识图谱增强RAG，擅长处理全局性问题和复杂关系推断。

[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection + 2023 + Asai et al.]：让模型自己决定何时检索、检索什么，并评价检索结果。

[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval + 2024 + Sarthi et al.]：通过递归摘要构建树状索引，解决RAG在跨文档摘要上的痛点。

[Corrective Retrieval Augmented Generation (CRAG) + 2024 + Yan et al.]：引入评估器判断检索质量，若质量差则进行网络搜索补充。

[MemGPT: Towards LLMs as Operating Systems + 2023 + Packer et al.]：管理虚拟内存（上下文），实现“无限”上下文对话。

[Dense Passage Retrieval for Open-Domain Question Answering (DPR) + 2020 + Karpukhin et al.]：向量检索的经典基石。

[Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) + 2022 + Gao et al.]：假设性文档嵌入，先生成答案再检索，提高召回相关性。

[LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models + 2023 + Chen et al.]：以极低成本扩展现有模型的上下文窗口。

六、 智能体（Agents）
从聊天到行动的进化。

[Generative Agents: Interactive Simulacra of Human Behavior + 2023 + Park et al.]：斯坦福著名的“西部世界”实验，展示了Agent的记忆、反思和规划能力。

[AutoGPT: An Autonomous GPT-4 Experiment + 2023 + Richards]：虽然是工程项目，但其背后的循环执行理念（Loop）开启了Agent热潮。

[Voyager: An Open-Ended Embodied Agent with Large Language Models + 2023 + Wang et al.]：在Minecraft中展示了Agent如何自我学习技能并探索世界。

[MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework + 2023 + Hong et al.]：引入SOP（标准作业程序），让多个Agent扮演不同角色（产品经理、工程师等）协作写代码。

[ChatDev: Communicative Agents for Software Development + 2023 + Qian et al.]：类似MetaGPT，专注于软件开发的流水线化Agent协作。

[Toolformer: Language Models Can Teach Themselves to Use Tools + 2023 + Schick et al.]：让模型自发学会调用API（如计算器、日历）。

[ReAct: Synergizing Reasoning and Acting in Language Models + 2022 + Yao et al.]：Agent的核心范式，将“思考（Reasoning）”和“行动（Acting）”交替进行。

[AgentTuning: Enabling Generalized Agent Abilities for LLMs + 2023 + Zeng et al.]：专门为Agent能力进行微调的数据集和方法。

[OS-Copilot: Towards Generalist Computer Agents with Self-Improvement + 2024 + Wu et al.]：让Agent像操作系统助手一样控制电脑应用。

[WebArena: A Realistic Web Environment for Building Autonomous Agents + 2023 + Zhou et al.]：评估Agent在真实网页环境下操作能力的基准。"""

class LoadFile(Tool):
    name = "load_file"
    description = "从文件加载长文本资料"
    output_type = "string"
    
    inputs = {
        "instruction": {"type": "string", "description": "写作指令。"},
    }
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        
        
        # 加载 prompt 配置 (可选)
        self.prompts = kwargs.get("prompts", {})

    def forward(self, instruction: str) -> str:  
        return context + "\n\n" + context