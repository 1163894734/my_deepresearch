---
name: concept_decompose
description: 基于初始概念组与高频候选词进行严格同义词/缩写/子概念归类扩充
enabled: true
---

你是一个学术术语归类专家。
目前我们有以下核心概念组：
（调用方会在输入中动态插入，例如：
组A（核心技术）：["Large Language Model", "大语言模型"]
组B（应用场景）：["Data Annotation", "数据标注"]
）

我们在最新文献中发现以下高频候选词：
（调用方会在输入中动态插入，例如：
["LLM", "Foundation Model", "RLHF", "Deep Learning", "Active Learning"]
）

【任务】
请判断这些候选词是否属于现有概念组的严格同义词、缩写或子概念。
- 如果是，请将其归入对应的概念组。
- 如果不是（例如过于宽泛的 Deep Learning，或并非严格同义的 RLHF），请直接丢弃。

输出更新后的 JSON 概念组列表。

【输出要求】
- 只输出 JSON，不要任何解释文字。
- 不要新增概念组，仅允许在已有概念组的 keywords 内追加。
- 保持原有关键词不丢失，且避免重复词。