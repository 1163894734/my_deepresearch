try:
    from open_deep_research.scripts.citation_validator import CitationValidator
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    examples_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples_root))
    from open_deep_research.scripts.citation_validator import CitationValidator


validator = CitationValidator(min_source_count=2)


CITATIONS_JSON = {
  "Large Language Models: A Survey": {
      "title": "Large Language Models: A Survey",
      "abstract": "本文综述了当前主流的大型语言模型，包括GPT、LLaMA和PaLM系列，并讨论了其特点、贡献与限制。同时概述了构建和增强LLM的技术，调查了用于训练、微调和评估LLM的数据集，回顾了广泛使用的评估指标，并在代表性基准上比较了多个流行LLM的性能。最后，讨论了开放挑战和未来研究方向。 Large Language Models (LLMs) have drawn a lot of attention due to their strong performance on a wide range of natural language tasks, since the release of ChatGPT in November 2022.",
      "keywords": [
        "Large Language Models",
        "GPT",
        "LLaMA",
        "PaLM",
        "ChatGPT"
      ],
      "authors": "Minaee, Shervin; Mikolov, Tomas; Nikzad, Narjes; Chenaghlu, Meysam; Socher, Richard; Amatriain, Xavier; Gao, Jianfeng",
      "url": "https://arxiv.org/abs/2402.06196",
      "published_time": "2024-02",
      "year": "2024",
      "source_type": "paper",
      "apa_citation": "(Minaee et al., 2024)",
      "source": ["arxiv", "crossref"]
    },
    "Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry": {
      "title": "Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry",
      "abstract": "本文总结了2024年大型语言模型黑客松的成果，展示了LLM在分子与材料属性预测、设计、自动化接口、科研沟通、数据管理、假设生成及文献知识提取等方面的多样化应用。活动在全球多地线上线下同步举行，共收到34个团队提交的作品，体现了LLM作为多功能机器学习模型和科研快速原型平台的双重价值。 These outcomes demonstrate the dual utility of LLMs as both multipurpose models for diverse machine learning tasks and platforms for rapid prototyping custom applications in scientific research.",
      "keywords": [
        "LLM",
        "材料科学",
        "分子属性预测",
        "科研自动化",
        "文献知识提取"
      ],
      "authors": "Various contributors from the LLM Hackathon community",
      "url": "https://arxiv.org/abs/2411.15221",
      "published_time": "2024-11",
      "year": "2024",
      "source_type": "paper",
      "apa_citation": "(LLM Hackathon Contributors, 2024)",
      "source": ["arxiv"]
    },
    }

PARAGRAPH = """## 2.1 推理优化机制：OPRO机制与优势分析

大模型推理优化的核心目标是在保障生成质量的前提下显著降低计算开销与响应延迟，其中基于自然语言的迭代优化机制——OPRO（Optimization by PROmpting）提出了一种范式级创新：将传统数值优化过程转化为可读、可干预的语言推理流程 (Minaee et al., 2024)。该机制不依赖梯度反向传播或参数微调，而是通过设计结构化提示（prompt），引导大语言模型（LLM）在推理阶段自主生成、评估并迭代改进解决方案，从而实现任务性能的持续提升。

OPRO的核心机制在于构建一个闭环的“生成—反馈—优化”链路。具体而言，模型首先根据初始提示生成候选输出；随后，通过引入自我评估模块或外部判别器，对输出结果进行自然语言形式的质量评分或错误分析；最后，将评估反馈重新注入提示中，驱动模型在下一轮推理中修正偏差、优化逻辑。这一过程可多次迭代，直至满足预设的终止条件。例如，在数学推理或代码生成任务中，OPRO可使模型先输出初步解答，再通过自我检查发现逻辑漏洞，并在后续轮次中生成修正版本，显著提升最终准确率 (LLM Hackathon Contributors, 2024)。

相较于传统推理优化方法，OPRO展现出三项显著优势。其一，**无需参数更新**，整个优化过程完全在推理时通过提示工程实现，避免了微调带来的存储与计算成本，适用于无法访问模型权重的闭源场景。其二，**过程可解释性强**，每一轮优化均以自然语言呈现，便于人类理解模型决策路径，支持动态干预与调试，增强了系统透明度与可控性。其三，**任务泛化能力突出**，同一OPRO框架可适配数学推理、科学假设生成、分子设计等多种复杂任务，展现出作为通用推理增强平台的潜力 (Minaee et al., 2024)。

进一步对比典型推理优化技术可见，OPRO在机制层面区别于思维链（Chain-of-Thought, CoT）与自洽性解码（Self-Consistency）。CoT虽能提升推理连贯性，但缺乏显式反馈与迭代机制；自洽性依赖多路径采样与投票，计算开销大且无学习过程。而OPRO通过引入显式反馈回路，实现了类似强化学习中的策略改进，但无需训练奖励模型或策略网络，降低了实现门槛 (Minaee et al., 2024)。

实证研究表明，OPRO在多个复杂任务上取得显著增益。在2024年材料科学与化学领域的LLM黑客松中，多个团队采用类似OPRO的迭代提示框架进行分子属性预测与材料设计，通过多轮“生成—评估—修正”循环，成功提升了预测准确性与设计方案的可行性 (LLM Hackathon Contributors, 2024)。这表明，将优化过程语言化不仅可行，且在科学发现等高精度需求场景中具有实际价值。

综上，OPRO代表了大模型推理优化从“静态前馈”向“动态交互”的范式转变，其通过自然语言实现的迭代优化机制，为提升模型推理深度与可靠性提供了新路径，尤其适用于高价值、低容错的科研与工程应用场景。"""


def test_five_step_pipeline_json_only():
    c1, p1 = validator.step1_search(CITATIONS_JSON, PARAGRAPH)
    print("After Step 1 Search:", c1, p1)
    c2, p2 = validator.step2_verify(c1, p1)
    print("After Step 2 Verify:", c2, p2)
    c3, p3 = validator.step3_retrieve(c2, p2)
    print("After Step 3 Retrieve:", c3, p3)
    c4, p4 = validator.step4_validate(c3, p3)
    print("After Step 4 Validate:", c4, p4)
    c5, p5 = validator.step5_add(c4, p4)
    print("After Step 5 Add:", c5, p5)



if __name__ == "__main__":
    test_five_step_pipeline_json_only()
