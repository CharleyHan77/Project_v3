# 创建文件: /workspace/Project_v3/diagnose_labels.py

import json
import numpy as np
import torch
from pareto_label import ParetoLabelGenerator
from collections import Counter, defaultdict

def diagnose_labels():
    """诊断Pareto标签的质量"""
    
    print("="*80)
    print("诊断Pareto标签质量")
    print("="*80)
    
    # 读取一个示例JSON
    json_path = "/workspace/Project_v3/init_validity_result_new/instance_0000_j10_m6_result.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    label_info = data['initialization_methods']
    methods = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 
               'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    
    # 显示原始性能数据
    print("\n1. 原始性能数据（4个目标，值越小越好）:")
    print("-"*80)
    print(f"{'方法':<15} {'makespan':<12} {'mean_flow':<12} {'max_load':<12} {'total_load':<12}")
    
    methods_objectives = {}
    for method in methods:
        obj = {
            'makespan': label_info[method]['makespan']['values']['mean'],
            'mean_flow_time': label_info[method]['mean_flow_time']['values']['mean'],
            'max_machine_load': label_info[method]['max_machine_load']['values']['mean'],
            'total_machine_load': label_info[method]['total_machine_load']['values']['mean']
        }
        methods_objectives[method] = obj
        print(f"{method:<15} {obj['makespan']:<12.1f} {obj['mean_flow_time']:<12.1f} "
              f"{obj['max_machine_load']:<12.1f} {obj['total_machine_load']:<12.1f}")
    
    # 测试不同策略
    print("\n" + "="*80)
    print("2. 不同Pareto策略的标签值和区分度")
    print("="*80)
    
    strategies = ['hypervolume', 'pareto_rank', 'domination_count', 'distance']
    generator = ParetoLabelGenerator()
    
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        print("-"*80)
        
        pareto_labels = generator.generate_labels(
            methods_objectives, 
            strategy=strategy,
            inverse_for_label=False
        )
        
        # 转为tensor（模拟数据集处理）
        label_values = torch.tensor([pareto_labels[method] for method in methods], dtype=torch.float)
        
        # 显示原始标签
        print(f"\n原始标签值:")
        for method in methods:
            print(f"  {method:<15}: {pareto_labels[method]:.6f}")
        
        # 统计信息
        values = list(pareto_labels.values())
        print(f"\n统计: 均值={np.mean(values):.6f}, 标准差={np.std(values):.6f}, "
              f"范围=[{np.min(values):.6f}, {np.max(values):.6f}]")
        
        # 检查唯一值数量
        unique_values = len(set(values))
        print(f"唯一标签数: {unique_values} / {len(methods)}")
        
        if unique_values < len(methods):
            print(f"⚠️  警告: 有 {len(methods) - unique_values} 个重复的标签值！")
            # 找出重复的
            value_counts = Counter(values)
            for val, count in value_counts.items():
                if count > 1:
                    duplicates = [m for m in methods if pareto_labels[m] == val]
                    print(f"   {count}个方法的标签都是 {val:.6f}: {', '.join(duplicates)}")
        
        # 测试softmax转换
        print(f"\n经过 softmax 后的分布:")
        softmax_probs = torch.softmax(label_values if strategy == 'hypervolume' else -label_values, dim=0)
        for i, method in enumerate(methods):
            print(f"  {method:<15}: {softmax_probs[i].item():.6f}")
        
        softmax_vals = softmax_probs.numpy()
        print(f"  Softmax分布统计: max={softmax_vals.max():.6f}, min={softmax_vals.min():.6f}, "
              f"差异={softmax_vals.max() - softmax_vals.min():.6f}")
        
        # 测试temperature softmax
        temperature = 0.05
        temp_softmax = torch.softmax(label_values / temperature if strategy == 'hypervolume' 
                                     else -label_values / temperature, dim=0)
        print(f"\n经过 temperature softmax (T={temperature}) 后的分布:")
        for i, method in enumerate(methods):
            print(f"  {method:<15}: {temp_softmax[i].item():.6f}")
        
        temp_vals = temp_softmax.numpy()
        print(f"  Temp-Softmax分布统计: max={temp_vals.max():.6f}, min={temp_vals.min():.6f}, "
              f"差异={temp_vals.max() - temp_vals.min():.6f}")
        
        # 找出最佳方法
        if strategy == 'hypervolume':
            best_idx = label_values.argmax().item()
        else:
            best_idx = label_values.argmin().item()
        
        print(f"\n最佳方法: {methods[best_idx]} (索引={best_idx})")
        print("="*80)


def analyze_dataset_distribution():
    """分析整个数据集的标签分布"""
    
    print("\n\n" + "="*80)
    print("3. 分析整个数据集的标签分布")
    print("="*80)
    
    import os
    label_root_path = "/workspace/Project_v3/init_validity_result_new"
    
    # 获取所有JSON文件
    json_files = []
    for file in os.listdir(label_root_path):
        if file.endswith('.json'):
            json_files.append(os.path.join(label_root_path, file))
    
    print(f"\n找到 {len(json_files)} 个实例")
    
    methods = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 
               'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    
    strategies = ['hypervolume', 'pareto_rank']
    
    for strategy in strategies:
        print(f"\n\n策略: {strategy}")
        print("-"*80)
        
        generator = ParetoLabelGenerator()
        
        # 统计最佳方法分布
        best_method_counts = Counter()
        label_uniqueness = []  # 记录每个实例有多少个唯一标签
        
        for json_file in json_files[:100]:  # 分析前100个
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                label_info = data['initialization_methods']
                
                # 构建objectives
                methods_objectives = {}
                for method in methods:
                    methods_objectives[method] = {
                        'makespan': label_info[method]['makespan']['values']['mean'],
                        'mean_flow_time': label_info[method]['mean_flow_time']['values']['mean'],
                        'max_machine_load': label_info[method]['max_machine_load']['values']['mean'],
                        'total_machine_load': label_info[method]['total_machine_load']['values']['mean']
                    }
                
                # 生成标签
                pareto_labels = generator.generate_labels(
                    methods_objectives, 
                    strategy=strategy,
                    inverse_for_label=False
                )
                
                # 检查唯一性
                values = list(pareto_labels.values())
                unique_count = len(set(values))
                label_uniqueness.append(unique_count)
                
                # 找出最佳方法
                if strategy == 'hypervolume':
                    best_method = max(pareto_labels.items(), key=lambda x: x[1])[0]
                else:
                    best_method = min(pareto_labels.items(), key=lambda x: x[1])[0]
                
                best_method_counts[best_method] += 1
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                continue
        
        # 显示结果
        print(f"\n最佳方法分布（前100个实例）:")
        total = sum(best_method_counts.values())
        for method, count in best_method_counts.most_common():
            percentage = 100.0 * count / total
            print(f"  {method:<15}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n标签唯一性统计:")
        print(f"  平均唯一标签数: {np.mean(label_uniqueness):.2f} / 8")
        print(f"  最少唯一标签数: {np.min(label_uniqueness)}")
        print(f"  最多唯一标签数: {np.max(label_uniqueness)}")
        
        if np.mean(label_uniqueness) < 6:
            print(f"\n⚠️⚠️⚠️  严重警告: 平均唯一标签数太低！")
            print(f"  这意味着很多方法有相同的标签值，模型难以区分")
            print(f"  建议: 尝试其他Pareto策略（如 pareto_rank 或 distance）")


def test_with_log_transform():
    """测试对数变换的影响"""
    
    print("\n\n" + "="*80)
    print("4. 测试对数变换的影响")
    print("="*80)
    
    json_path = "/workspace/Project_v3/init_validity_result_new/instance_0000_j10_m6_result.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    label_info = data['initialization_methods']
    methods = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 
               'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    
    methods_objectives = {}
    for method in methods:
        methods_objectives[method] = {
            'makespan': label_info[method]['makespan']['values']['mean'],
            'mean_flow_time': label_info[method]['mean_flow_time']['values']['mean'],
            'max_machine_load': label_info[method]['max_machine_load']['values']['mean'],
            'total_machine_load': label_info[method]['total_machine_load']['values']['mean']
        }
    
    generator = ParetoLabelGenerator()
    strategy = 'hypervolume'
    
    pareto_labels = generator.generate_labels(
        methods_objectives, 
        strategy=strategy,
        inverse_for_label=False
    )
    
    label_values = torch.tensor([pareto_labels[method] for method in methods], dtype=torch.float)
    
    print(f"\n原始标签值:")
    for i, method in enumerate(methods):
        print(f"  {method:<15}: {label_values[i].item():.6f}")
    
    # 对数变换（模拟dataset.py中的处理）
    log_label_values = torch.log(label_values + 1)
    
    print(f"\n经过 log(x+1) 变换后:")
    for i, method in enumerate(methods):
        print(f"  {method:<15}: {log_label_values[i].item():.6f}")
    
    print(f"\n对数变换后的统计:")
    print(f"  原始: 范围=[{label_values.min():.6f}, {label_values.max():.6f}], "
          f"标准差={label_values.std():.6f}")
    print(f"  变换后: 范围=[{log_label_values.min():.6f}, {log_label_values.max():.6f}], "
          f"标准差={log_label_values.std():.6f}")
    
    # 测试softmax
    print(f"\n变换后的 temperature softmax (T=0.05):")
    temp_softmax = torch.softmax(log_label_values / 0.05, dim=0)
    for i, method in enumerate(methods):
        print(f"  {method:<15}: {temp_softmax[i].item():.6f}")


if __name__ == "__main__":
    diagnose_labels()
    analyze_dataset_distribution()
    test_with_log_transform()