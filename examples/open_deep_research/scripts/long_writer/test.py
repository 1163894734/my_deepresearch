from collections import Counter
import json
initial_concepts=json.loads("""[
  {
    "concept_name": "大模型",
    "keywords": ["大语言模型", "LLM", "Large Language Model"]
  }
]""")
response_json = json.loads("""[{"Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models": {"authors": "Anonymous", "year": "2026", "title": "Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models", "url": "https://arxiv.org/abs/2603.13985", "source_type": "paper", "apa_citation": "(Anonymous, 2026)", "key_words": ["agent","agent","LLM", "Post-Training", "Supervised Fine-Tuning", "Reinforcement Learning", "Hybrid Methods", "SFT", "RL", "Fine-Tuning", "Training Paradigms", "Generalization"], "abstract": "Pre-trained Large Language Model (LLM) exhibits broad capabilities, yet, for specific tasks or domains their attainment of higher accuracy and more reliable reasoning generally depends on post-training through Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Although often treated as distinct methodologies, recent theoretical and empirical developments demonstrate that SFT and RL are closely connected. This study presents a comprehensive and unified perspective on LLM post-training with SFT and RL. We first provide an in-depth overview of both techniques, examining their objectives, algorithmic structures, and data requirements. We then systematically analyze their interplay, highlighting frameworks that integrate SFT and RL, hybrid training pipelines, and methods that leverage their complementary strengths. Drawing on a representative set of recent application studies from 2023 to 2025, we identify emerging trends, characterize the rapid shift toward hybrid post-training paradigms, and distill key takeaways that clarify when and why each method is most effective. By synthesizing theoretical insights, practical methodologies, and empirical evidence, this study establishes a coherent understanding of SFT and RL within a unified framework and outlines promising directions for future research in scalable, efficient, and generalizable LLM post-training."}}]""")
existing_keywords = set()

for concept in initial_concepts:
    # 提取关键词列表，过滤空值，添加到集合
    keywords = [str(kw).strip() for kw in concept.get("keywords", []) if str(kw).strip()]
    existing_keywords.update(keywords)
print(f"🔍 已有关键词（{len(existing_keywords)} 个）: {existing_keywords}")

all_keywords = []
for paper_dict in response_json:  # response 是列表，每个元素是字典
    for paper_title, paper_info in paper_dict.items():  # 遍历每个字典的键值对
        keywords = paper_info.get("key_words", [])
        if isinstance(keywords, list):
            all_keywords.extend(keywords)
print(f"🔍 从检索结果中提取到 {len(all_keywords)} 个关键词: {all_keywords}")
# 统计频次
term_counter = Counter(all_keywords)

# 提取频次 >=2 且不在 existing_keywords 中的关键词
candidate_words = [term for term, count in term_counter.items() 
    if count >= 2 and term not in existing_keywords]

candidate_words = [term for term, count in term_counter.items() if count >= 2 and term not in existing_keywords]
print(f"🔍 提取到 {len(candidate_words)} 个候选关键词: {candidate_words[:20]}...")
print("candidate_words:", candidate_words)