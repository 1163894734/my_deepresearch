# from __future__ import annotations

# import json
# import os
# import re
# import time
# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import dataclass
# from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

# from smolagents import ToolCallingAgent
# from smolagents.agents import ActionOutput, ToolOutput
# from smolagents.memory import ActionStep
# from smolagents.models import ChatMessage, MessageRole
# from smolagents.monitoring import LogLevel
# from FlagEmbedding import FlagReranker
# from utils.common_utils import safe_json_parse

# # 引入您的学术检索服务（请确保路径与您的项目结构匹配）
# try:
#     from long_writer.academic_search_service import AcademicSearchService
# except ImportError:
#     from .long_writer.academic_search_service import AcademicSearchService

# @dataclass
# class RetrievedBlock:
#     source: str
#     text: str

# class InterpretationAgent(ToolCallingAgent):
#     """术语解读工作流：大小模型协同 + BGE重排 + 动态大纲 + 严谨裁判兜底 + 学术API检索。"""

#     def __init__(self, model, small_model=None, tools: Optional[List] = None, **kwargs):
#         self.small_model = small_model if small_model is not None else model
#         super().__init__(model=model, tools=tools or [], **kwargs)
        
#         self.run_dir = f"outputs/interpretation_{time.strftime('%Y%m%d_%H%M%S')}"
#         os.makedirs(self.run_dir, exist_ok=True)

#         self.local_search_tool_name = "coarse_rag"  
#         self.web_search_tool_name = "web_search"    
#         self.rerank_top_k = 5
#         self.judge_rounds = 2 
        
#         self.chunk_size = 500
#         self.chunk_overlap = 100
        
#         # 初始化学术搜索相关的 state（AcademicSearchService 会读取这些配置）
#         if self.state is None:
#             self.state = {}
#         self.state.setdefault("search_engine", "arxiv") # 默认使用 arxiv
#         self.state.setdefault("search_sort", "date")    
        
#         self.logger.log("🛠️ 正在加载本地 BGE-Reranker 模型...", level=LogLevel.INFO)
#         self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

#     def _step_stream(self, memory_step: ActionStep) -> Generator[ToolOutput | ActionOutput, None, None]:
#         del memory_step
#         start_time = time.time()
#         try:
#             self.logger.log_rule("📘 术语解读流程启动 (融合学术 API 模式)", level=LogLevel.INFO)
#             raw_task = str(self.task or "").strip()

#             # 1) 提取名词
#             term = self._extract_tech_term(raw_task)
#             self.logger.log(f"✅ 核心术语提取完成: [{term}]", level=LogLevel.INFO)

#             # 2) 检索与重排 (三源并行: Local + Web + Academic)
#             self.logger.log("🌐 启动多源并行检索 (本地 + 网页 + 学术)...", level=LogLevel.INFO)
#             local_blocks, web_blocks, academic_blocks = self._parallel_retrieve(term, raw_task)
            
#             merged = self._deduplicate_blocks(local_blocks + web_blocks + academic_blocks)
#             self.logger.log(f"📥 原始汇总块数: {len(merged)} (其中学术库贡献 {len(academic_blocks)} 块)", level=LogLevel.INFO)
            
#             top_blocks = self._rerank_blocks(merged, query=f"{term} {raw_task}", top_k=self.rerank_top_k)
#             self.logger.log(f"🔝 BGE 重排选出最优 {len(top_blocks)} 个证据块", level=LogLevel.INFO)
            
#             # 3) 抽取客观事实
#             self.logger.log("📝 正在合成客观事实底稿 (小模型模式)...", level=LogLevel.INFO)
#             objective_facts = self._synthesize_objective_facts(term, top_blocks)

#             # 4) 生成标准大纲
#             self.logger.log("📋 正在构建动态评估大纲 (小模型模式)...", level=LogLevel.INFO)
#             outline = self._build_dynamic_outline(term, objective_facts)

#             # 5) 三个写手并行起稿
#             styles = {
#                 "百科版": "角色：辞海编辑。风格：学术、严谨、客观。结构：定义、起源、原理、应用、结论。",
#                 "专报版": "角色：智库分析师。风格：战略、前瞻、精炼。重点：行业趋势、挑战、对策建议。",
#                 "科普版": "角色：科普作家。风格：生动、通俗、形象。要求：多用比喻，严禁公式。",
#             }
#             self.logger.log("✍️ 大模型并行撰写三版稿件中...", level=LogLevel.INFO)
#             drafts = self._generate_three_drafts(term, objective_facts, outline, styles)

#             # 6) 裁判审修 (并行优化)
#             self.logger.log(f"⚖️ 进入裁判审核模式 (共 {self.judge_rounds} 轮循环，并行校对中)...", level=LogLevel.INFO)
#             judged = {}

#             with ThreadPoolExecutor(max_workers=3) as ex:
#                 # 将三个校对任务同时提交到线程池
#                 fut_map = {
#                     name: ex.submit(self._judge_and_refine, term, outline, objective_facts, text, self.judge_rounds)
#                     for name, text in drafts.items()
#                 }
                
#                 # 收集并行校对的结果
#                 for name, fut in fut_map.items():
#                     self.logger.log(f"🔍 正在等待校对结果: {name}", level=LogLevel.INFO)
#                     try:
#                         judged[name] = fut.result()
#                         self.logger.log(f"✅ 校对完成: {name}", level=LogLevel.INFO)
#                     except Exception as e:
#                         self.logger.log(f"⚠️ {name} 校对过程发生异常: {e}，回退至初稿", level=LogLevel.ERROR)
#                         judged[name] = drafts[name] # 兜底逻辑：如果裁判报错，直接使用初稿

#             # 7) 组装输出
#             output = self._assemble_output(term, objective_facts, top_blocks, outline, judged)
            
#             duration = time.time() - start_time
#             self.logger.log(f"🚀 任务全部完成，总耗时: {duration:.2f}s", level=LogLevel.INFO)
            
#             self.state["interpretation_result"] = output

#             # ========== 修改：将解读报告保存至时间戳文件夹 ==========
#             import os
#             # 过滤一下 term 里的特殊字符，作为合法的文件名
#             safe_term = "".join([c for c in term if c.isalnum() or c in [' ', '_']]).strip().replace(' ', '_')
#             output_file = os.path.join(self.run_dir, f"interpretation_report_{safe_term}.md")
            
#             with open(output_file, "w", encoding="utf-8") as f:
#                 f.write(output)
#             self.logger.log(f"💾 解读报告已保存至: {output_file}", level=LogLevel.INFO)
#             # ========================================================

#             yield ActionOutput(output=output, is_final_answer=True)
            
#         except Exception as exc:
#             self.logger.log(f"❌ InterpretationAgent 发生致命错误: {exc}", level=LogLevel.ERROR)
#             raise

#     # ============== 检索核心逻辑 ==============

#     def _parallel_retrieve(self, term: str, raw_task: str) -> Tuple[List[RetrievedBlock], List[RetrievedBlock], List[RetrievedBlock]]:
#         """三路并行检索引擎"""
#         with ThreadPoolExecutor(max_workers=3) as ex:
#             fut_local = ex.submit(self._retrieve_local, term, raw_task)
#             fut_web = ex.submit(self._retrieve_web, term, raw_task)
#             fut_academic = ex.submit(self._retrieve_academic, term, raw_task)
            
#             return fut_local.result(), fut_web.result(), fut_academic.result()

#     def _retrieve_academic(self, term: str, raw_task: str) -> List[RetrievedBlock]:
#         """调用学术 API 服务进行深度检索"""
#         try:
#             self.logger.log("🎓 启动学术引擎辅助检索...", level=LogLevel.INFO)
#             query_context = f"{term} {raw_task}"
            
#             # 1. 解析英文学术关键词
#             intent = AcademicSearchService.parse_search_intent(self.model, query_context)
#             search_query = intent.get("search_query", term)
            
#             # 2. 执行检索
#             papers = AcademicSearchService.search_academic_papers(
#                 self, 
#                 search_query=search_query, 
#                 year_start=intent.get("year_start", ""), 
#                 year_end=intent.get("year_end", ""), 
#                 max_fetch=10, 
#                 target_count=3 # 为了速度，取Top-3的学术摘要即可
#             )
            
#             if not papers:
#                 return []
                
#             # 3. 将论文摘要拼接并分块
#             combined_text = ""
#             for title, info in papers.items():
#                 combined_text += f"【学术论文】{title} ({info.get('year', '')})\n作者: {info.get('authors', '')}\n摘要: {info.get('abstract', '')}\n\n"
                
#             return self._result_to_blocks(combined_text, source="api:academic_search")
            
#         except Exception as e:
#             self.logger.log(f"⚠️ 学术检索过程发生异常: {e}", level=LogLevel.ERROR)
#             return []

#     def _retrieve_local(self, term: str, raw_task: str) -> List[RetrievedBlock]:
#         query = f"{term} {raw_task}"
#         args = {"query": query}
#         try:
#             result = self.execute_tool_call(self.local_search_tool_name, args)
#             return self._result_to_blocks(result, source=f"local:{self.local_search_tool_name}")
#         except: return []

#     def _retrieve_web(self, term: str, raw_task: str) -> List[RetrievedBlock]:
#         query = f"{term} 原理 应用 趋势"
#         try:
#             result = self.execute_tool_call(self.web_search_tool_name, {"query": query})
#             return self._result_to_blocks(result, source="web_search")
#         except: return []

#     def _result_to_blocks(self, tool_result: Any, source: str) -> List[RetrievedBlock]:
#         text = str(tool_result or "").strip()
#         if not text: return []
        
#         # 顺手修复换行符残留问题
#         text = re.sub(r'(?<!\n)\n(?!\n)', '', text) 
        
#         sentences = re.split(r'(?<=[。！？!?\n])', text)
#         blocks, current = [], ""
#         for s in sentences:
#             if len(current) + len(s) > self.chunk_size:
#                 blocks.append(RetrievedBlock(source=source, text=current.strip()))
#                 current = s
#             else: current += s
#         if current: blocks.append(RetrievedBlock(source=source, text=current.strip()))
#         return blocks

#     def _deduplicate_blocks(self, blocks: Sequence[RetrievedBlock]) -> List[RetrievedBlock]:
#         seen, out = set(), []
#         for b in blocks:
#             norm = re.sub(r"\W+", "", b.text.lower())[:100]
#             if norm not in seen:
#                 seen.add(norm); out.append(b)
#         return out

#     def _rerank_blocks(self, blocks: Sequence[RetrievedBlock], query: str, top_k: int = 5) -> List[RetrievedBlock]:
#         if len(blocks) <= top_k: return list(blocks)
#         try:
#             scores = self.reranker.compute_score([[query, b.text] for b in blocks])
#             scored = sorted(zip(blocks, scores), key=lambda x: x[1], reverse=True)
#             return [b for b, s in scored[:top_k]]
#         except: return list(blocks)[:top_k]

#     # ============== 内容生成核心逻辑 ==============

#     def _synthesize_objective_facts(self, term: str, blocks: Sequence[RetrievedBlock]) -> str:
#         snippets = "\n".join(f"[{i+1}] {b.text}" for i, b in enumerate(blocks))
#         prompt = f"归纳事实底稿（客观、无评价）。名词：{term}\n资料：{snippets}"
#         return self._call_model_text(prompt, temperature=0.1, use_small_model=True)

#     def _build_dynamic_outline(self, term: str, facts: str) -> str:
#         prompt = f"为【{term}】生成评估大纲（3-5个维度）。\n事实：{facts}"
#         return self._call_model_text(prompt, temperature=0.1, use_small_model=True)

#     def _generate_three_drafts(self, term: str, facts: str, outline: str, styles: Dict[str, str]) -> Dict[str, str]:
#         outputs = {}
#         with ThreadPoolExecutor(max_workers=3) as ex:
#             fut_map = {name: ex.submit(self._write_style_draft, term, facts, outline, sp) for name, sp in styles.items()}
#             for name, fut in fut_map.items():
#                 outputs[name] = fut.result() if fut.exception() is None else "（生成失败）"
#         return outputs

#     def _write_style_draft(self, term: str, facts: str, outline: str, style_prompt: str) -> str:
#         prompt = f"""
# {style_prompt}
# 请基于以下【客观事实】撰写【{term}】的解读稿。
# 要求：
# 1. 必须严格遵循【参考大纲】的结构。
# 2. 内容长度控制在 800 字左右。
# 3. 不要包含“以下是修订版”之类的废话，直接输出正文。

# 【参考大纲】：
# {outline}

# 【客观事实】：
# {facts}
# """
#         return self._call_model_text(prompt, temperature=0.5, use_small_model=False)

#     def _judge_and_refine(self, term: str, outline: str, facts: str, draft: str, rounds: int = 2) -> str:
#         current = draft
#         for i in range(rounds):
#             judge_prompt = f"""
# 你是一名严厉的编辑。请根据以下大纲和事实检查稿件。
# 【大纲】: {outline}
# 【事实】: {facts}
# 【稿件】: {current}

# 检查标准：1. 事实是否准确 2. 是否完全覆盖大纲维度 3. 是否有重复废话。
# 请直接输出 JSON，格式：{{"pass": bool, "issues": [], "revision_instruction": ""}}
# """
#             raw_judge = self._call_model_text(judge_prompt, temperature=0.0, use_small_model=False)
#             data = safe_json_parse(raw_judge)
            
#             # ========== 新增：在控制台稍微透传一下模型到底返回了啥 ==========
#             # 只截取前 100 个字符，避免控制台被刷屏
#             self.logger.log(f"🧠 [裁判原生输出预览]: {raw_judge[:100].replace(chr(10), ' ')}...", level=LogLevel.INFO)
#             self.logger.log(f"🧩 [JSON 强制解析结果]: {data}", level=LogLevel.INFO)
#             # ================================================================
            
#             if data.get("pass") is True:
#                 self.logger.log(f"✅ 第 {i+1} 轮校对通过", level=LogLevel.INFO)
#                 return current

#             instruction = data.get("revision_instruction", "请进一步优化逻辑并对齐大纲。")
#             self.logger.log(f"⚠️ 第 {i+1} 轮校对未通过，建议修订方向: {instruction[:50]}...", level=LogLevel.INFO)
            
#             revise_prompt = f"请根据修订建议优化稿件。保持原风格，直接输出修订后的正文。\n建议：{instruction}\n原稿：{current}"
#             current = self._call_model_text(revise_prompt, temperature=0.3, use_small_model=False)

#         return current

#     def _assemble_output(self, term: str, objective_facts: str, blocks: Sequence[RetrievedBlock], outline: str, drafts: Dict[str, str]) -> str:
#         clean_drafts = {}
#         for k, v in drafts.items():
#             # 安全地清理模型可能在其输出开头生成的标题词
#             clean_text = re.sub(r"^(#+ *(百科版|专报版|科普版|解读报告)).*\n?", "", v.strip(), flags=re.IGNORECASE).strip()
#             clean_drafts[k] = clean_text or v

#         src_lines = [f"- [{i+1}] ({b.source}) {b.text[:80]}..." for i, b in enumerate(blocks)]
#         parts = [
#             f"# 技术名词解读：{term}", 
#             "\n## 1. 动态评估大纲", outline,
#             "\n## 2. 客观事实底稿", objective_facts,
#             "\n## 3. 证据参考 (Top 5)", "\n".join(src_lines) if src_lines else "（无可用检索结果）",
#             "\n## 4. 多维深度解读",
#             "### [百科版]", clean_drafts.get("百科版", "（缺失）"),
#             "\n### [专报版]", clean_drafts.get("专报版", "（缺失）"),
#             "\n### [科普版]", clean_drafts.get("科普版", "（缺失）"),
#         ]
#         return "\n".join(parts)

#     def _call_model_text(self, prompt: str, temperature: float = 0.2, use_small_model: bool = False) -> str:
#         messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
#         target_model = self.small_model if use_small_model else self.model
#         resp = target_model(messages, temperature=temperature)
#         result_text = str(getattr(resp, "content", "") or "").strip()

#         # ========== 新增：将详细输入输出写入本地文件 ==========
#         model_type = "小模型" if use_small_model else "大模型"
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#         log_file = os.path.join(self.run_dir, "llm_trace_log.md")
        
#         try:
#             with open(log_file, "a", encoding="utf-8") as f:
#                 f.write(f"\n\n## 🕒 [{timestamp}] | 🤖 调用模型: {model_type} | 🌡️ Temp: {temperature}\n")
#                 f.write("### 📥 [INPUT PROMPT]\n```text\n" + prompt + "\n```\n")
#                 f.write("### 📤 [MODEL OUTPUT]\n```text\n" + result_text + "\n```\n")
#                 f.write("---\n")
#         except Exception as e:
#             self.logger.log(f"⚠️ 写入日志文件失败: {e}", level=LogLevel.ERROR)
#         # ======================================================

#         return result_text

#     def _extract_tech_term(self, user_input: str) -> str:
#         prompt = f"请从输入中提取核心技术名词。只输出名词。输入：{user_input}"
#         term = self._call_model_text(prompt, temperature=0.0, use_small_model=True)
#         return re.sub(r"[\n\r]+", " ", term).strip(" ：:。.!?\"'")


# __all__ = ["InterpretationAgent"]