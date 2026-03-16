from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from smolagents import ToolCallingAgent
from smolagents.agents import ActionOutput, ToolOutput
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import LogLevel


@dataclass
class RetrievedBlock:
	source: str
	text: str


class InterpretationAgent(ToolCallingAgent):
	"""术语解读智能体：并行检索 + 客观事实抽取 + 三风格写作 + 裁判反思。"""

	def __init__(self, model, tools: Optional[List] = None, **kwargs):
		super().__init__(model=model, tools=tools or [], **kwargs)
		self.local_search_tool_name = "coarse_rag"  # 本地向量库检索工具（优先）
		self.web_search_tool_name = "web_search"    # Web 搜索工具
		self.rerank_top_k = 5
		self.judge_rounds = 3

	def _step_stream(self, memory_step: ActionStep) -> Generator[ToolOutput | ActionOutput, None, None]:
		del memory_step
		try:
			self.logger.log_rule("📘 术语解读流程启动", level=LogLevel.INFO)

			raw_task = str(self.task or "").strip()
			if not raw_task:
				raise ValueError("任务为空")

			term = self._extract_tech_term(raw_task)
			self.logger.log(f"🔎 提取技术名词: {term}", level=LogLevel.INFO)

			# 2) 并行双路检索：本地向量库 + Web
			local_blocks, web_blocks = self._parallel_retrieve(term, raw_task)

			# 3) 去重 + 重排 top-5 + 客观事实
			merged = self._deduplicate_blocks(local_blocks + web_blocks)
			top_blocks = self._rerank_blocks(merged, query=f"{term}\n{raw_task}", top_k=self.rerank_top_k)
			objective_facts = self._synthesize_objective_facts(term, top_blocks)

			# 4) 三个写手并行起稿
			styles = {
				"百科版": "你的角色是《辞海》编辑，语言客观、条理清晰，多用名词解释。",
				"专报版": "你的角色是智库分析师，重点分析产业背景、应用潜力和未来趋势，给领导看的。",
				"科普版": "你的角色是科普大V，必须把技术比作生活中常见事物，不要用难懂公式。",
			}
			drafts = self._generate_three_drafts(term, objective_facts, styles)

			# 5) 裁判 agent：标准大纲核对 + 3轮反思修订
			outline = self._build_standard_outline(term)
			judged = {
				name: self._judge_and_refine(term, outline, objective_facts, text, rounds=self.judge_rounds)
				for name, text in drafts.items()
			}

			# 6) 装订展示
			output = self._assemble_output(term, objective_facts, top_blocks, judged)
			self.state["interpretation_result"] = output
			yield ActionOutput(output=output, is_final_answer=True)
		except Exception as exc:
			self.logger.log(f"❌ InterpretationAgent 错误: {exc}", level=LogLevel.ERROR)
			raise

	# ============== LLM helpers ==============

	def _call_model_text(self, prompt: str, temperature: float = 0.2) -> str:
		messages = [
			ChatMessage(
				role=MessageRole.USER,
				content=[{"type": "text", "text": prompt}],
			)
		]
		resp = self.model(messages, temperature=temperature)
		return str(getattr(resp, "content", "") or "").strip()

	def _extract_tech_term(self, user_input: str) -> str:
		prompt = (
			"请从用户输入中提取一个最核心的技术名词。"
			"只输出名词本身，不要解释，不要加标点。\n\n"
			f"用户输入：{user_input}"
		)
		term = self._call_model_text(prompt, temperature=0.0)
		term = re.sub(r"[\n\r]+", " ", term).strip(" ：:。.!?\"'")
		if not term:
			term = user_input[:40]
		return term

	# ============== Retrieval ==============

	def _parallel_retrieve(self, term: str, raw_task: str) -> Tuple[List[RetrievedBlock], List[RetrievedBlock]]:
		with ThreadPoolExecutor(max_workers=2) as ex:
			f_local = ex.submit(self._retrieve_local, term, raw_task)
			f_web = ex.submit(self._retrieve_web, term, raw_task)
			local_blocks = f_local.result() if f_local else []
			web_blocks = f_web.result() if f_web else []
		return local_blocks, web_blocks

	def _retrieve_local(self, term: str, raw_task: str) -> List[RetrievedBlock]:
		query = f"{term}\n{raw_task}"
		tool_names = [self.local_search_tool_name, "fine_rag"]
		for name in tool_names:
			for args in ({"input": query}, {"query": query}, {"task": query}, {"text": query}):
				try:
					result = self.execute_tool_call(name, args)
					blocks = self._result_to_blocks(result, source=f"local:{name}")
					if blocks:
						return blocks
				except Exception:
					continue
		return []

	def _retrieve_web(self, term: str, raw_task: str) -> List[RetrievedBlock]:
		query = f"{term} 技术原理 应用 趋势\n{raw_task}"
		for args in ({"query": query}, {"input": query}, {"task": query}, {"text": query}):
			try:
				result = self.execute_tool_call(self.web_search_tool_name, args)
				blocks = self._result_to_blocks(result, source=f"web:{self.web_search_tool_name}")
				if blocks:
					return blocks
			except Exception:
				continue
		return []

	def _result_to_blocks(self, tool_result: Any, source: str) -> List[RetrievedBlock]:
		text = str(tool_result or "").strip()
		if not text:
			return []

		# 尝试按常见结构切块
		raw_parts = re.split(r"\n\s*\n+|\n[-*]\s+|\n\d+[\).、]\s+", text)
		blocks: List[RetrievedBlock] = []
		for part in raw_parts:
			p = re.sub(r"\s+", " ", part).strip()
			if len(p) >= 40:
				blocks.append(RetrievedBlock(source=source, text=p))

		if not blocks:
			blocks = [RetrievedBlock(source=source, text=text[:3000])]
		return blocks

	# ============== Rerank + Fact synthesis ==============

	def _deduplicate_blocks(self, blocks: Sequence[RetrievedBlock]) -> List[RetrievedBlock]:
		seen = set()
		out: List[RetrievedBlock] = []
		for b in blocks:
			norm = re.sub(r"\W+", "", b.text.lower())[:300]
			if not norm or norm in seen:
				continue
			seen.add(norm)
			out.append(b)
		return out

	def _rerank_blocks(self, blocks: Sequence[RetrievedBlock], query: str, top_k: int = 5) -> List[RetrievedBlock]:
		if not blocks:
			return []
		q_tokens = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", query.lower()) if t]

		def score(text: str) -> float:
			t = text.lower()
			hits = sum(1 for tok in q_tokens if tok and tok in t)
			density = hits / max(1, len(q_tokens))
			length_penalty = 1.0 if 80 <= len(text) <= 1200 else 0.85
			return density * length_penalty

		ranked = sorted(blocks, key=lambda x: score(x.text), reverse=True)
		return ranked[:max(1, top_k)]

	def _synthesize_objective_facts(self, term: str, blocks: Sequence[RetrievedBlock]) -> str:
		snippets = "\n\n".join(f"[{i+1}] {b.text}" for i, b in enumerate(blocks))
		prompt = f"""
你是技术事实整理员。请基于给定资料写一段不带情绪色彩、只陈述事实的客观材料。
要求：
1) 只基于资料，不要臆测；
2) 结构清晰：定义/原理、关键技术点、应用现状、局限性；
3) 400~700字。

技术名词：{term}

资料：
{snippets}
""".strip()
		return self._call_model_text(prompt, temperature=0.1)

	# ============== Writing ==============

	def _generate_three_drafts(self, term: str, facts: str, styles: Dict[str, str]) -> Dict[str, str]:
		outputs: Dict[str, str] = {}
		with ThreadPoolExecutor(max_workers=3) as ex:
			fut_map = {
				name: ex.submit(self._write_style_draft, term, facts, style_prompt)
				for name, style_prompt in styles.items()
			}
			for name, fut in fut_map.items():
				try:
					outputs[name] = fut.result()
				except Exception:
					outputs[name] = "（生成失败）"
		return outputs

	def _write_style_draft(self, term: str, facts: str, style_prompt: str) -> str:
		prompt = f"""
{style_prompt}

请基于以下客观事实材料，撰写“{term}”的解读稿。
要求：
- 不得偏离事实；
- 字数 500~900 字；
- 结构完整、可直接发布。

客观事实材料：
{facts}
""".strip()
		return self._call_model_text(prompt, temperature=0.3)

	# ============== Judge & Reflection ==============

	def _build_standard_outline(self, term: str) -> str:
		return (
			f"标准大纲（{term}）:\n"
			"1. 概念定义\n"
			"2. 核心原理/关键组成\n"
			"3. 主要应用场景\n"
			"4. 当前限制与风险\n"
			"5. 未来趋势（基于证据，不夸张）"
		)

	def _judge_and_refine(self, term: str, outline: str, facts: str, draft: str, rounds: int = 3) -> str:
		current = draft
		for i in range(rounds):
			judge_prompt = f"""
你是裁判编辑。请检查下列初稿是否存在：
1) 扭曲原意/事实不一致；
2) 与标准大纲缺项；
3) 夸张或情绪化描述。

请只输出 JSON：
{{
  "pass": true/false,
  "issues": ["..."],
  "revision_instruction": "..."
}}

技术名词：{term}

标准大纲：
{outline}

事实材料：
{facts}

当前稿件：
{current}
""".strip()
			raw = self._call_model_text(judge_prompt, temperature=0.0)
			data = self._safe_json_parse(raw)
			if bool(data.get("pass", False)):
				return current

			instruction = str(data.get("revision_instruction", "")).strip()
			issues = data.get("issues", [])
			issues_txt = "\n".join(f"- {x}" for x in issues if isinstance(x, str))

			revise_prompt = f"""
请按以下审校意见修订稿件，保持原有风格，但必须忠实事实并补齐大纲缺项。

审校意见：
{issues_txt or instruction}

标准大纲：
{outline}

事实材料：
{facts}

原稿：
{current}

输出修订后完整稿件。
""".strip()
			current = self._call_model_text(revise_prompt, temperature=0.2)

			self.logger.log(f"🧪 裁判反思轮次 {i+1}/{rounds} 完成", level=LogLevel.INFO)
		return current

	# ============== Output ==============

	def _assemble_output(
		self,
		term: str,
		objective_facts: str,
		blocks: Sequence[RetrievedBlock],
		drafts: Dict[str, str],
	) -> str:
		src_lines = [f"- [{i+1}] ({b.source}) {b.text[:180]}..." for i, b in enumerate(blocks)]

		parts = [
			f"# 技术名词解读：{term}",
			"",
			"## 客观事实底稿",
			objective_facts,
			"",
			"## 证据块（重排 Top-5）",
			"\n".join(src_lines) if src_lines else "（无可用检索结果）",
			"",
			"## 百科版",
			drafts.get("百科版", "（缺失）"),
			"",
			"## 专报版",
			drafts.get("专报版", "（缺失）"),
			"",
			"## 科普版",
			drafts.get("科普版", "（缺失）"),
		]
		return "\n".join(parts)

	@staticmethod
	def _safe_json_parse(text: str) -> Dict[str, Any]:
		raw = str(text or "").strip()
		if not raw:
			return {}
		try:
			data = json.loads(raw)
			return data if isinstance(data, dict) else {}
		except Exception:
			m = re.search(r"\{.*\}", raw, re.DOTALL)
			if not m:
				return {}
			try:
				data = json.loads(m.group(0))
				return data if isinstance(data, dict) else {}
			except Exception:
				return {}


__all__ = ["InterpretationAgent", "RetrievedBlock"]

