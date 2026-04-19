---
name: get_finalization_sop
description: 当你接受长文本报告生成任务的【最终阶段】时，必须首先阅读此 SOP。严格按照这些步骤编写代码，汇总所有内容、生成摘要、格式化参考文献、输出最终 Markdown 报告。
enabled: true
type: sop
---

# 【长文本最终阶段标准作业程序 (Finalization SOP)】

你需要在一个 Python 沙盒环境中，严格按照以下步骤编写代码并调用工具：

## 【环境变量注入】
- `written_sections`: 章节阶段返回的所有已完成章节 Dict
- `full_content`: 章节阶段返回的拼接后完整正文
- `all_citations_used`: 章节阶段返回的所有使用过的引用 Dict
- `cleaned_task`: 用户的原始任务
- `outline`: 规划阶段的最终大纲字符串

## 【核心流程 - 5步】

### 第 1 步：校验与整理已完成的章节
1. 验证 `written_sections` 中的所有章节都已成功生成
2. 检查 `full_content` 是否为非空字符串且长度 >= 2000 字
3. 获取章节总数：`total_sections = len(written_sections)`
4. 创建章节有序列表：`section_list = list(written_sections.keys())`

### 第 2 步：生成摘要
1. 调用 `write_abstract(full_text=full_content, available_citations=all_citations_used, task=cleaned_task)`
2. 返回值是字典：`{abstract_content: str, key_findings: List[str], metadata: {...}}`
3. 保存摘要：`abstract_text = 返回值['abstract_content']`（应为 200-300 字）
4. 如果摘要生成失败，使用降级方案：取 `full_content` 的前 300 字作为备选摘要

### 第 3 步：格式化参考文献列表
1. 对 `all_citations_used` 进行去重（按 title 键唯一）
2. 调用 `format_citations(citations_dict=all_citations_used, style='APA')`
3. 返回值是字典：`{references_text: str, formatted_citations: List[Dict], count: int}`
4. 保存格式化参考文献：`references_text = 返回值['references_text']`
5. 记录总引用数：`total_citations = 返回值['count']`
6. 按 APA 标准排序：作者姓氏 A-Z，相同作者按发表年份升序排列

### 第 4 步：组装最终 Markdown 文档
1. 构造完整 Markdown，按以下顺序拼接：
   ```
   # {从 cleaned_task 中提取的标题或 "长文本研究报告"}
   
   ## 📋 摘要 (Abstract)
   {abstract_text}
   
   ---
   
   {full_content}
   
   ---
   
   ## 📚 参考文献 (References)
   {references_text}
   ```

2. 保存到变量：`final_markdown = 上述构造的字符串`
3. 验证 Markdown 有效性（包含所有章节标题、摘要、参考文献）

### 第 5 步：返回最终结果
调用 `final_answer()` 返回以下结构：

```python
{
    "status": "success",  # 或 "failure"（若存在严重错误）
    "final_markdown": final_markdown,  # 完整的可输出 Markdown 字符串
    "finalization_metadata": {
        "abstract_word_count": len(abstract_text.split()),
        "total_sections": total_sections,
        "total_citations": total_citations,
        "full_content_word_count": len(full_content.split()),
        "outline_provided": bool(outline),
        "generation_timestamp": datetime.now().isoformat()
    }
}
```

## 【关键约束】

1. **摘要质量**：
   - 必须基于 full_content（整个正文）生成，而非大纲
   - 长度 200-300 字，涵盖核心问题、关键发现、研究意义
   - 如摘要生成失败，降级使用正文前 300 字

2. **参考文献管理**：
   - 必须对 all_citations_used 进行完全去重（无重复）
   - 所有引用必须按 APA 格式规范
   - 引用总数 >= 3（及格线），>= 5（良好）

3. **Markdown 结构**：
   - 标题层级清晰（# 报告标题 → ## 摘要、章节、参考文献）
   - 摘要必须在所有正文章节之前
   - 参考文献必须在最后

4. **内容完整性**：
   - 返回的 Markdown 必须包含：摘要 + full_content + 参考文献
   - full_content 必须包含所有已生成的章节
   - 总字数（摘要 + 正文，不含参考文献）>= 2000 字

5. **错误处理**：
   - 摘要生成失败 → 使用降级方案（正文前 300 字）
   - 参考文献格式化失败 → 返回原始 Dict 格式（标记为降级）
   - 任何阶段失败都应记录错误，尽可能返回部分结果

## 【成功标准】（供系统验证）

- ✅ 摘要长度 200-300 字，内容完整
- ✅ 参考文献 >= 5 条，格式正确（APA），无查重复
- ✅ 最终 Markdown 包含所有章节、摘要、参考文献
- ✅ 总字数（摘要 + 正文，不含参考文献）>= 2000 字
- ✅ Markdown 格式规范，可直接用编辑器或发布平台显示