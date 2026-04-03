import argparse
import json
import sys
from pathlib import Path

# 导入所有组件，确保在项目根目录运行不受路径问题困扰
from scripts.long_writer.cli_debugger import run_component_cli
from scripts.long_writer.component_keyword_search import KeywordSearchExpansionComponent
from scripts.long_writer.component_outline import OutlineGenerationComponent
from scripts.long_writer.component_introduction import IntroductionWritingComponent
from scripts.long_writer.component_body import BodyWritingComponent
from scripts.long_writer.component_abstract import AbstractWritingComponent
from scripts.long_writer.component_conclusion import ConclusionWritingComponent
from scripts.long_writer.component_references import ReferencesWritingComponent
from scripts.long_writer.component_academic_search import AcademicSearchComponent

def get_shared_citations():
    """提取多个组件共用的文献测试数据"""
    return {
        "Large Language Models: A Survey": {
            "title": "Large Language Models: A Survey",
            "abstract": "本文综述了当前主流的大型语言模型，包括GPT、LLaMA和PaLM系列...",
            "keywords": ["Large Language Models", "GPT", "LLaMA", "PaLM", "ChatGPT"],
            "authors": "Minaee, Shervin; Mikolov, Tomas; Nikzad, Narjes; Chenaghlu, Meysam; Socher, Richard; Amatriain, Xavier; Gao, Jianfeng",
            "url": "https://arxiv.org/abs/2402.06196",
            "published_time": "2024-02",
            "year": "2024",
            "source_type": "paper",
            "apa_citation": "(Minaee et al., 2024)"
        },
        "Reflections from the 2024 Large Language Model (LLM) Hackathon...": {
            "title": "Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry",
            "abstract": "本文总结了2024年大型语言模型黑客松的成果...",
            "keywords": ["LLM", "材料科学", "分子属性预测", "科研自动化", "文献知识提取"],
            "authors": "Various contributors from the LLM Hackathon community",
            "url": "https://arxiv.org/abs/2411.15221",
            "published_time": "2024-11",
            "year": "2024",
            "source_type": "paper",
            "apa_citation": "(LLM Hackathon Contributors, 2024)"
        }
    }

def load_mock_data(component_name: str) -> dict:
    """集中管理所有组件的测试载荷 (Payload)"""
    workspace_root = Path(__file__).resolve().parent
    
    # 辅助读取可能存在的文件内容
    pre_section_path = workspace_root / "pre_section.md"
    long_text_path = workspace_root / "long_text.md"
    prev_section_content = pre_section_path.read_text(encoding="utf-8") if pre_section_path.exists() else "（前一段内容的测试占位符...）"
    full_text_content = long_text_path.read_text(encoding="utf-8") if long_text_path.exists() else "（全文内容的测试占位符...）"

    if component_name == "keyword_search":
        return {"task": "大模型领域前沿进展"}
        
    elif component_name == "academic_search":
        return {"task": "大模型领域，只看最近2年的论文"}
        
    elif component_name == "outline":
        return {
            "query": "大模型领域前沿进展",
            "available_citations": {
                "OPUS: Optimizer-induced Projected Utility Selection": {
                    "authors": "阿里巴巴, 上海交大, UW-Madison等",
                    "year": "2026",
                    "title": "OPUS: Optimizer-induced Projected Utility Selection for Dynamic Data Selection in LLM Pre-training",
                    "url": "https://arxiv.org/pdf/2602.0540",
                    "source_type": "paper",
                    "apa_citation": "(阿里巴巴等, 2026)",
                    "key_words": ["动态数据选择", "预训练优化", "AdamW", "Muon", "效用最大化", "Bench-Proxy"]
                }
            }
        }
        
    elif component_name == "intro":
        return {
            "input": {
                "section": {"title": "引言", "goal": "交代背景与问题", "word_count_target": 600},
                "available_citations": get_shared_citations(),
                "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
            }
        }
        
    elif component_name == "body":
        return {
            "input": {
                "section": {"title": "2.1 推理优化机制", "goal": "分析OPRO机制与优势", "word_count_target": 900},
                "fine_rag_context": "OPRO 将优化过程转化为自然语言迭代过程。",
                "available_citations": get_shared_citations(),
                "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。",
                "prev_section_content": prev_section_content,
            }
        }
        
    elif component_name == "abstract":
        return {
            "input": {
                "section": {"title": "摘要", "goal": "凝练核心发现", "word_count_target": 300},
                "conclusion_text": "（结论内容）",
                "full_text": full_text_content,
                "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。",
            }
        }
        
    elif component_name == "conclusion":
        return {
            "input": {
                "section": {"title": "结论", "goal": "总结贡献与展望", "word_count_target": 500},
                "available_citations": get_shared_citations(),
                "full_text": full_text_content,
                "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。",
            }
        }
        
    elif component_name == "references":
        return {
            "input": {
                "section": {"title": "参考文献"},
                # 注意：References组件当时测试里给的是 List 格式，这里做适配
                "available_citations": list(get_shared_citations().values()),
                "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。"
            }
        }
        
    return {}

def main():
    # 组件名字映射到类实例
    COMPONENT_MAP = {
        "keyword_search": KeywordSearchExpansionComponent(),
        "academic_search": AcademicSearchComponent(),
        "outline": OutlineGenerationComponent(),
        "intro": IntroductionWritingComponent(),
        "body": BodyWritingComponent(),
        "abstract": AbstractWritingComponent(),
        "conclusion": ConclusionWritingComponent(),
        "references": ReferencesWritingComponent(),
    }

    parser = argparse.ArgumentParser(description="Long Writer 各组件独立测试脚本")
    parser.add_argument(
        "-c", "--component", 
        type=str, 
        required=True,
        choices=list(COMPONENT_MAP.keys()),
        help="指定要测试的组件简写名称"
    )
    args = parser.parse_args()

    target_component = COMPONENT_MAP[args.component]
    mock_payload = load_mock_data(args.component)
    
    print(f"\n=============================================")
    print(f"🚀 开始测试组件: {target_component.name}")
    print(f"=============================================\n")
    
    # 将字典转换为 JSON 字符串
    payload_json_text = json.dumps(mock_payload, ensure_ascii=False)
    
    # 巧妙复用你已经写好的 cli_debugger 工具进行测试
    try:
        sys.exit(run_component_cli(
            target_component,
            argv=["--input-json", payload_json_text]
        ))
    except SystemExit as e:
        # run_component_cli 返回的是状态码，我们正常捕获并透传
        sys.exit(e.code)
    except Exception as e:
        print(f"\n❌ 测试过程发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()