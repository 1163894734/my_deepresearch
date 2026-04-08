import concurrent.futures
from typing import List, Dict
from smolagents import Tool

# 如果你本地的 utils.common_utils 中有 safe_json_parse，请保持这个 import
# 如果没有，可以直接在这个文件里实现那个安全的 JSON 解析函数
from utils.common_utils import safe_json_parse 

def build_core_text(doc: Dict, max_chars: int = 1400) -> str:
    """提取核心文本，替代原来的 CoreTextExtractor"""
    parts = []
    if doc.get("title"): parts.append(f"[标题] {doc['title'].strip()}")
    if doc.get("abstract"): parts.append(f"[摘要] {doc['abstract'].strip()}")
    if doc.get("independent_claim"): parts.append(f"[独立权利要求] {doc['independent_claim'].strip()}")
    if doc.get("introduction_or_conclusion"): parts.append(f"[引言/结论] {doc['introduction_or_conclusion'].strip()}")
    if not parts and doc.get("text"): parts.append(doc["text"].strip())

    merged = "\n".join(parts).strip()
    return merged if len(merged) <= max_chars else merged[:max_chars] + "..."

class FilterDocumentsTool(Tool):
    name = "filter_documents"
    description = "第一步：对输入的原始文献进行核心文本提取，并判断其是否属于目标领域，剔除无关文献。"
    
    inputs = {
        "documents": {"type": "array", "description": "原始文献字典列表(须包含 title, abstract, text 等)"},
        "domain": {"type": "string", "description": "目标前沿领域名称"},
        "min_prob": {"type": "number", "description": "保留文献的最低置信度阈值，默认 0.6", "nullable": True}
    }
    output_type = "array"

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.batch_size = 5
        self.max_workers = 10

    def _judge_batch(self, batch_docs: List[Dict], domain: str) -> List[float]:
        texts_str = "\n".join([f"[{i}] {doc['core_text'][:600]}..." for i, doc in enumerate(batch_docs)])
        base_prompt = (
            f"你是一个严格的学术文献领域分类器。请判断以下 {len(batch_docs)} 篇文献是否属于【{domain}】领域。\n"
            f"请给出一个 0.0 到 1.0 之间的置信度分数。\n"
            f"【要求】严格返回纯 JSON 数组，长度必须刚好为 {len(batch_docs)}。示例：[0.95, 0.12, 0.88]\n\n"
            f"文献列表：\n{texts_str}"
        )

        prompt = base_prompt
        for attempt in range(3):
            try:
                res = self.model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
                scores = safe_json_parse(res.content, fallback_type=list)
                if isinstance(scores, list) and len(scores) == len(batch_docs):
                    return [float(s) for s in scores]
            except Exception as e:
                print(f"⚠️ 解析失败 (尝试 {attempt+1}/3): {e}")
            prompt = base_prompt + f"\n\n【警告】必须输出长度为 {len(batch_docs)} 的纯数字 JSON 数组！"

        return [0.5] * len(batch_docs)

    def forward(self, documents: List[Dict], domain: str, min_prob: float = 0.6) -> List[Dict]:
        print(f"\n📥 [Filter Skill] 启动瘦身与判别，处理 {len(documents)} 篇文献...")
        
        # 1. 瘦身
        for doc in documents:
            doc["core_text"] = build_core_text(doc)

        # 2. 并发判别
        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        all_scores = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._judge_batch, batch, domain) for batch in batches]
            for future in futures:
                all_scores.extend(future.result())
        
        # 3. 过滤
        kept_docs = []
        for doc, score in zip(documents, all_scores):
            if score >= min_prob:
                doc["domain_probability"] = score
                kept_docs.append(doc)
                
        print(f"📤 [Filter Skill] 过滤完成 -> 保留: {len(kept_docs)} 篇 | 剔除: {len(documents) - len(kept_docs)} 篇")
        return kept_docs