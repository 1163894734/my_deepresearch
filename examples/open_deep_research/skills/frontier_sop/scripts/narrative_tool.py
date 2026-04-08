import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
from smolagents import Tool
from utils.common_utils import safe_json_parse

def parse_year(ts: str) -> int:
    if not ts: return None
    ts = str(ts).strip()
    m = re.search(r"(19|20)\d{2}", ts)
    if m: return int(m.group(0))
    try: return datetime.fromisoformat(ts).year
    except: return None

def count_by_year(docs: List[Dict]) -> Dict[int, int]:
    c = defaultdict(int)
    for d in docs:
        y = parse_year(d.get("timestamp", ""))
        if y is not None: c[y] += 1
    return dict(c)

def compute_cagr(yearly_count: Dict[int, int]) -> float:
    if not yearly_count: return 0.0
    years = sorted(yearly_count)
    if len(years) < 2: return 0.0
    periods = years[-1] - years[0]
    if periods <= 0: return 0.0
    start_val, end_val = max(1, yearly_count.get(years[0], 0)), max(0, yearly_count.get(years[-1], 0))
    if end_val <= 0: return -1.0
    return (end_val / start_val) ** (1 / periods) - 1

class ExtractNarrativeTool(Tool):
    name = "extract_narrative"
    description = "第三步：针对一个聚类簇，提取其前沿技术名词、内涵，并计算技术趋势（CAGR等）。"
    
    inputs = {
        "cluster_data": {"type": "any", "description": "单个聚类簇的字典数据"},
        "domain": {"type": "string", "description": "所属领域"},
        "all_domain_docs": {"type": "array", "description": "所有被保留的全局文献，用于计算相对趋势占比"}
    }
    output_type = "any"

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, cluster_data: Dict, domain: str, all_domain_docs: List[Dict]) -> Dict:
        cid = cluster_data.get("cluster_id")
        print(f"\n📥 [Narrative Skill] 正在处理簇 [{cid}]...")
        
        # 1. 提取文本内涵
        summaries = "\n\n".join(cluster_data.get("representative_summaries", []))
        prompt = (
            f"目标领域：{domain}\n"
            "请严格以JSON格式输出该技术簇的核心信息，包含：\n"
            "1. technology_term: 前沿技术名词（具象且有高辨识度）\n"
            "2. technology_problem: 技术解决的具体痛点\n"
            "3. technology_method: 采用的核心方法\n"
            "4. application_direction: 主要应用场景\n\n"
            "在提取技术痛点、核心方法和应用方向时，无论原始文献是什么语言，你返回的 JSON/字典内容必须全部翻译并总结为初始任务使用的语言。"
            f"摘要集合：\n{summaries}"
        )
        
        res = self.model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        data = safe_json_parse(res.content, fallback_type=dict)
        
        cluster_data.update({
            "technology_term": data.get("technology_term", f"{domain}-候选技术_{cid}"),
            "technology_problem": data.get("technology_problem", "未提取成功"),
            "technology_method": data.get("technology_method", "未提取成功"),
            "application_direction": data.get("application_direction", "未提取成功")
        })
        
        # 2. 趋势分析
        cluster_docs = cluster_data.pop("cluster_docs", []) # 拿出来顺便删除，防止臃肿
        cluster_yearly = count_by_year(cluster_docs)
        domain_yearly = count_by_year(all_domain_docs)
        
        years = sorted(set(cluster_yearly) | set(domain_yearly))
        yearly_share = {y: (cluster_yearly.get(y, 0) / domain_yearly.get(y, 0)) if domain_yearly.get(y, 0) > 0 else 0.0 for y in years}
        
        burst_detected, burst_reason = False, "未检测到显著突现"
        if len(years) >= 3:
            y2, y1 = years[-2], years[-1]
            delta_share = yearly_share.get(y1, 0.0) - yearly_share.get(y2, 0.0)
            if delta_share >= 0.03 and cluster_yearly.get(y1, 0) >= cluster_yearly.get(y2, 0):
                burst_detected = True
                burst_reason = f"近两年占比提升 {delta_share:.2%}，判定为技术突现"

        cluster_data["trend"] = {
            "yearly_count": dict(sorted(cluster_yearly.items())),
            "yearly_share": dict(sorted(yearly_share.items())),
            "cagr": compute_cagr(cluster_yearly),
            "burst_detected": burst_detected,
            "burst_reason": burst_reason
        }
        return cluster_data