<!-- ---
name: keywords_search_sop
description: 当你接受【文献搜索】时，必须首先阅读此 SOP。严格按照这些步骤编写代码，汇总所有内容、生成摘要、格式化参考文献、输出最终 Markdown 报告。
enabled: true
type: sop
---
【文献调研标准工作流 (SOP)】

当用户需要了解某个领域的研究情况时，你必须遵循以下步骤：
1. **检索阶段 (Search)**：使用 `academic_search` 工具，传入核心英文关键词、时间限制及排序要求（如需要高影响力文章，选用 engine='openalex', sort_by='citation'）。
2. **筛选阶段 (Filter)**：分析搜索返回的论文列表标题和年份，挑选出 1-2 篇最贴合用户需求的核心论文。
3. **精读阶段 (Deep Read)**：获取该论文的 URL。
   - 如果是 arXiv 链接，将其转换为 pdf 链接（将 abs 替换为 pdf）。
   - 使用 `visit_page` 工具直接访问该 PDF URL，提取论文的全文内容。
4. **总结阶段 (Summarize)**：结合论文全文的关键信息（如核心创新点、实验数据等），用中文向用户输出最终的高质量综述。 -->