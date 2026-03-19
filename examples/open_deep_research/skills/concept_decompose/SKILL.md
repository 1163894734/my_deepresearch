---
name: concept_decompose
description: 基于初始概念组与高频候选词进行严格同义词/缩写/子概念归类扩充
enabled: true
---

任务：更新现有的概念分组JSON数据

【初始数据】格式：
[
  {
    "concept_name": "概念名称",
    "keywords": ["词1", "词2"]
  }
]

【新增关键词】格式：
["LLM", "Foundation Model", "RLHF", "Deep Learning", "Active Learning"]

【更新要求】
1. 分析新增关键词的语义含义，判断它们应该：
   - 归入现有的某个概念组
   - 或者创建新的概念组

2. 对于每个新增关键词：
   - 如果与现有概念组中的关键词语义相近（同义词、相关术语），则加入该组
   - 如果无法归入现有任何一组，则创建新的概念组

3. 语义分析原则，示例：
   - "LLM"和"Foundation Model"都是大型语言模型/基础模型相关
   - "RLHF"是强化学习人类反馈，与模型训练相关
   - "Deep Learning"是机器学习的子领域
   - "Active Learning"是机器学习的一种方法

4. 概念组命名要准确概括组内关键词的共性

5. 输出格式必须严格保持与初始数据相同的JSON格式：
[
  {
    "concept_name": "概念名称",
    "keywords": ["关键词1", "关键词2", ...]
  },
  ...
]

请开始更新，只输出JSON结果，不要有其他解释。