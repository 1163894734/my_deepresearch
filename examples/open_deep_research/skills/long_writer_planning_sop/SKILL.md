---
name: get_planning_sop
description: 当你接受长文本报告生成任务的【规划阶段】时，必须首先阅读此 SOP。严格按照这些步骤使用工具，生成最终大纲与全局文献库。
enabled: false
type: sop
---

# 【长文本规划阶段标准作业程序 (Planning SOP)】

你需要在一个 Python 沙盒环境中，严格按照以下步骤编写代码并调用工具：

## 【环境变量注入】
- `cleaned_task`: 用户的原始任务描述（已清理）
- `outline_score_threshold`: 大纲通过的最低分数（默认 90）
- `outline_max_iter`: 最多允许多少轮大纲修订（默认 2）

## 【核心流程 - 5步】

### 第 1 步：初始关键词挖掘与第一轮文献检索
1. 使用 `keywords_search_result = json_load(keywords_search_agent(task="【领域文献搜索】请针对用户的需求搜索最相关的材料，以下是用户的原始任务：" + cleaned_task))` 获取初始文献
2. 得到的`keywords_search_result`是一个字典，包含：
   - `available_citations`: Dict[文献标题 -> {authors, year, url, abstract, ...}]
   - `candidate_keywords`: List[候选关键词字符串]
3. 将 `available_citations` 保存为 `global_citations = 返回值['available_citations']`
4. 记录初始候选关键词：`initial_keywords = 返回值['candidate_keywords']`

### 第 2 步：大纲生成
1. 调用 `outline_generation(input="{task=" + cleaned_task + ", available_citations=" + json.dumps(global_citations) + "}")`
2. 返回值是一个字符串，包含：
   - `outline_v1`: 初步大纲文本
   - `available_citations`: 可能返回新增文献（Dict 格式）
3. 保存初步大纲：`current_outline = 返回值['outline_v1']`
4. 更新全局文献库：`global_citations.update(返回值.get('available_citations', {}))`

### 第 3 步：大纲反思与迭代修订（最多进行 outline_max_iter 轮）
对于 `i` 在 `range(outline_max_iter)` 中：

#### 3.1 调用反思评分工具
1. 调用 `outline_reflection(input="大纲:\n" + current_outline + "\n参考资料:\n" + json.dumps(list(global_citations.keys())[:5]))`
2. 返回值是 JSON，结构：`{score: int (0-100), feedback: str, is_pass: bool}`
3. 解析得分：`score = 返回值['score']`，反馈：`feedback = 返回值['feedback']`

#### 3.2 判断是否通过
- 若 `score >= outline_score_threshold` 或 `is_pass == True`：
  - **立即停止循环，跳到第 4 步**
  - 不再执行修订，返回当前大纲
- 若 `score < outline_score_threshold` 且仍有修订次数：
  - **继续执行 3.3 步**

#### 3.3 修订大纲（仅当分数不足时）
1. 调用 `outline_revision(input="当前大纲:\n" + current_outline + "\n反思反馈:\n" + feedback + "\n参考资料:\n" + json.dumps(list(global_citations.keys())[:5]))`
2. 返回值是修订后的大纲字符串
3. 更新：`current_outline = 返回值`
4. 累计修订次数：`revision_count += 1`
5. **继续下一轮循环**

### 第 4 步：返回最终规划结果
调用 `final_answer()` 返回以下结构的字典（既返回给系统，也作为 final_answer 的参数）：

```python
{
    "outline": current_outline,  # 最终大纲字符串
    "available_citations": global_citations,  # 全局参考文献库 { title -> {authors, year, url, ...} }
    "planning_metadata": {
        "initial_keywords": initial_keywords,
        "final_score": score,  # 最后一次反思得分
        "revision_count": revision_count,  # 实际修订轮数
        "citation_count": len(global_citations)  # 文献总数
    }
}
```

## 【关键约束】

1. **文献库去重**：
   - `global_citations` 必须是 Dict 格式，键为文献标题（唯一）
   - 不允许重复的标题键

2. **迭代上限**：
   - 严格不能超过 `outline_max_iter` 轮修订
   - 达到阈值分数后必须立即停止

3. **错误恢复**：
   - 若任何工具调用异常，返回当前最佳结果（当前 outline + global_citations）
   - 不中断流程

4. **日志透明**：
   - 每轮反思的得分、反馈、修订意见都应在代码中 print 或写日志

## 【成功标准】（供系统验证）

- ✅ 返回的大纲包含 3-8 个清晰主要章节
- ✅ 全局文献库 >= 5 条高质量文献
- ✅ 最终得分 >= 90 或已达 outline_max_iter 轮修订上限
- ✅ 返回的 outline 为非空字符串，长度 >= 200 字