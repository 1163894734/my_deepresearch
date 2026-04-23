import os
import json
import time

# 直接导入工具类
from scripts.frontier_tools import (
    MultiSourceDataCollectorTool,
    FragmentedInfoAggregatorTool,
    KeyFrontierTechnologyMiningTool,
    WeakSignalTechnologyMiningTool,
    EvolutionPathAnalysisTool,
    KeyTechnologyEvolutionSensingTool
)

def main():
    target_topic = "3D heterogeneous integration"
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"unit_test_{run_timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开启工具链逐个单元测试 | 主题: {target_topic}\n")

    # ==========================================
    # 工具 1: 多源数据采集
    # ==========================================
    print("="*60)
    print("🛠️ [测试 1/6] MultiSourceDataCollectorTool")
    print("="*60)
    collector = MultiSourceDataCollectorTool()
    # 直接调用 forward 方法，绕过大模型解析
    raw_data = collector.forward(domain=target_topic, max_items_per_source=15)
    
    print(f"\n✅ 采集结果类型: {type(raw_data)}")
    for source, items in raw_data.items():
        print(f"   - {source} 来源获取了 {len(items)} 条数据")
    if raw_data.get("paper"):
        print(f"   📄 第一篇论文数据样例:\n{json.dumps(raw_data['paper'][0], ensure_ascii=False, indent=2)}")


    # ==========================================
    # 工具 2: 碎片化信息聚合
    # ==========================================
    print("\n" + "="*60)
    print("🛠️ [测试 2/6] FragmentedInfoAggregatorTool")
    print("="*60)
    aggregator = FragmentedInfoAggregatorTool()
    agg_result = aggregator.forward(
        multi_source_data=raw_data, 
        output_formats=["json"], 
        out_dir=out_dir, 
        file_stem="01_structured"
    )
    structured_table = agg_result["table"]
    
    print(f"\n✅ 聚合表总长度: {agg_result['summary']['total_records']}")
    print(f"✅ 来源分布: {agg_result['summary']['by_source']}")
    print(f"📊 聚合表第一行提取出的字段:\n{json.dumps(structured_table[0] if structured_table else {}, ensure_ascii=False, indent=2)}")


    # ==========================================
    # 工具 3: 关键前沿技术挖掘
    # ==========================================
    print("\n" + "="*60)
    print("🛠️ [测试 3/6] KeyFrontierTechnologyMiningTool")
    print("="*60)
    frontier_miner = KeyFrontierTechnologyMiningTool()
    frontier_result = frontier_miner.forward(
        structured_table=structured_table,
        domain_keywords=[target_topic],
        output_format="dict",
        out_dir=out_dir,
        file_stem="02_frontier"
    )
    
    print(f"\n✅ 发现前沿技术数量: {frontier_result['summary']['frontier_count']} / 总簇数: {frontier_result['summary']['total_clusters']}")
    if frontier_result.get("frontier_technologies"):
        print("🔥 提炼出的第一个前沿技术内涵:")
        first_frontier = frontier_result["frontier_technologies"][0]
        print(json.dumps(first_frontier['analysis'], ensure_ascii=False, indent=2))


    # ==========================================
    # 工具 4: 弱信号技术识别
    # ==========================================
    print("\n" + "="*60)
    print("🛠️ [测试 4/6] WeakSignalTechnologyMiningTool")
    print("="*60)
    weak_miner = WeakSignalTechnologyMiningTool()
    weak_result = weak_miner.forward(
        structured_table=structured_table,
        domain_keywords=[target_topic],
        output_format="dict",
        out_dir=out_dir,
        file_stem="03_weak"
    )
    
    print(f"\n✅ 发现弱信号技术数量: {weak_result['summary']['weak_signal_count']}")
    if weak_result.get("weak_signal_technologies"):
        print("🌱 提炼出的第一个弱信号内涵:")
        first_weak = weak_result["weak_signal_technologies"][0]
        print(json.dumps(first_weak['analysis'], ensure_ascii=False, indent=2))


    # ==========================================
    # 工具 5: 演进路径分析
    # ==========================================
    print("\n" + "="*60)
    print("🛠️ [测试 5/6] EvolutionPathAnalysisTool")
    print("="*60)
    evo_analyzer = EvolutionPathAnalysisTool()
    evo_result = evo_analyzer.forward(
        frontier_list=frontier_result,
        weak_signal_list=weak_result,
        structured_table=structured_table,
        output_format="dict",
        out_dir=out_dir,
        file_stem="04_evo"
    )
    
    print(f"\n✅ 主流路线数量: {evo_result['summary']['mainstream_route_count']}")
    print(f"📖 技术概述:\n{evo_result['technical_overview']}")
    if evo_result.get("mainstream_routes"):
        print(f"\n🛤️ 第一条路线细节: {evo_result['mainstream_routes'][0]['route_name']}")
        print(f"   - 优势: {evo_result['mainstream_routes'][0]['advantages'][:2]}")
        print(f"   - 瓶颈: {evo_result['mainstream_routes'][0]['bottlenecks'][:2]}")


    # ==========================================
    # 工具 6: 重点技术演进态势感知
    # ==========================================
    print("\n" + "="*60)
    print("🛠️ [测试 6/6] KeyTechnologyEvolutionSensingTool")
    print("="*60)
    sensing_analyzer = KeyTechnologyEvolutionSensingTool()
    sensing_result = sensing_analyzer.forward(
        frontier_list=frontier_result,
        weak_signal_list=weak_result,
        structured_table=structured_table,
        technology_name=target_topic,
        output_format="dict",
        out_dir=out_dir,
        file_stem="05_sensing"
    )
    
    print(f"\n✅ 总体研判结论:\n{sensing_result['overall_conclusion']}")
    print(f"\n👀 各路线成熟度与趋势:")
    for route in sensing_result.get('route_sensing', []):
        print(f"   - 【{route['route_name']}】: {route['maturity_stage']} | {route['future_trend']}")

    print(f"\n🎉 测试全部完成！原始落盘文件见: {out_dir}")

if __name__ == "__main__":
    main()