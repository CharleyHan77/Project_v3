#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标签分布分析脚本
功能：输出不同标签（最佳初始化方法）对应的fjs实例名称
与 Project_v3/dataset.py 中的打标签规则一致
"""

import os
import json
import numpy as np
from collections import defaultdict

# 8种初始化方法名称
METHOD_NAMES = [
    "FIFO_SPT",
    "FIFO_EET",
    "MOPNR_SPT",
    "MOPNR_EET",
    "LWKR_SPT",
    "LWKR_EET",
    "MWKR_SPT",
    "MWKR_EET"
]

def get_files(path):
    """递归获取目录下所有文件"""
    rsp = []
    for p in os.scandir(path):
        if os.path.isdir(p):
            rsp += get_files(p)
        else:
            rsp.append(p.path)
    return rsp

def get_best_method(label_info, label_name="mean"):
    """
    获取最佳初始化方法（与dataset.py中的规则一致）
    返回: 0-7的索引，对应8种初始化方法
    """
    makespans = np.array([
        label_info["FIFO_SPT"]["makespan"]["values"][label_name],
        label_info["FIFO_EET"]["makespan"]["values"][label_name],
        label_info["MOPNR_SPT"]["makespan"]["values"][label_name],
        label_info["MOPNR_EET"]["makespan"]["values"][label_name],
        label_info["LWKR_SPT"]["makespan"]["values"][label_name],
        label_info["LWKR_EET"]["makespan"]["values"][label_name],
        label_info["MWKR_SPT"]["makespan"]["values"][label_name],
        label_info["MWKR_EET"]["makespan"]["values"][label_name]
    ])
    # 取log后再argmin（与dataset.py第43-74行一致）
    log_makespans = np.log(makespans + 1)
    return np.argmin(log_makespans)

def analyze_label_distribution(label_root_path, label_name="mean"):
    """
    分析标签分布
    返回: 每个标签对应的实例列表和统计信息
    """
    # 获取所有label文件
    label_files = get_files(label_root_path)
    label_files = [f for f in label_files if f.endswith('.json')]
    
    print(f"找到 {len(label_files)} 个标签文件\n")
    
    # 统计每个标签对应的实例
    label_to_instances = defaultdict(list)
    instance_details = []
    
    for label_path in label_files:
        try:
            with open(label_path, "r") as f:
                label = json.load(f)
            
            # 兼容新旧两种格式（与dataset.py第26-38行一致）
            if "instance_info" in label:
                # 新格式
                instance_name = label["instance_info"]["file_name"]
                dataset_name = label["instance_info"].get("dataset", "unknown")
            else:
                # 旧格式
                instance_name = label["instance"]
                dataset_name = label.get("dataset", "unknown")
            
            label_info = label["initialization_methods"]
            best_method_idx = get_best_method(label_info, label_name)
            best_method_name = METHOD_NAMES[best_method_idx]
            
            # 记录详细信息
            makespans = [
                label_info[method]["makespan"]["values"][label_name]
                for method in METHOD_NAMES
            ]
            
            detail = {
                "instance_name": instance_name,
                "dataset": dataset_name,
                "best_method_idx": best_method_idx,
                "best_method_name": best_method_name,
                "best_makespan": makespans[best_method_idx],
                "all_makespans": dict(zip(METHOD_NAMES, makespans))
            }
            
            label_to_instances[best_method_idx].append(detail)
            instance_details.append(detail)
            
        except Exception as e:
            print(f"处理文件 {label_path} 时出错: {e}")
            continue
    
    return label_to_instances, instance_details

def print_report(label_to_instances, instance_details):
    """打印分析报告"""
    total_instances = len(instance_details)
    
    print("=" * 80)
    print("数据标签分布分析报告")
    print("=" * 80)
    print(f"\n总实例数: {total_instances}\n")
    
    print("-" * 80)
    print("标签分布统计:")
    print("-" * 80)
    
    for method_idx in range(8):
        instances = label_to_instances[method_idx]
        count = len(instances)
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        
        print(f"\n标签 {method_idx}: {METHOD_NAMES[method_idx]}")
        print(f"  数量: {count} ({percentage:.2f}%)")
        
        if count > 0:
            # 按数据集分组
            dataset_groups = defaultdict(list)
            for inst in instances:
                dataset_groups[inst["dataset"]].append(inst["instance_name"])
            
            print(f"  涉及数据集: {len(dataset_groups)} 个")
            for dataset, inst_list in sorted(dataset_groups.items()):
                print(f"    - {dataset}: {len(inst_list)} 个实例")
    
    print("\n" + "=" * 80)
    print("详细实例列表（按标签分组）")
    print("=" * 80)
    
    for method_idx in range(8):
        instances = label_to_instances[method_idx]
        if len(instances) == 0:
            continue
            
        print(f"\n{'='*80}")
        print(f"标签 {method_idx}: {METHOD_NAMES[method_idx]} - {len(instances)} 个实例")
        print(f"{'='*80}")
        
        # 按数据集分组显示
        dataset_groups = defaultdict(list)
        for inst in instances:
            dataset_groups[inst["dataset"]].append(inst)
        
        for dataset, inst_list in sorted(dataset_groups.items()):
            print(f"\n数据集: {dataset} ({len(inst_list)} 个实例)")
            print("-" * 80)
            
            # 按实例名称排序
            inst_list.sort(key=lambda x: x["instance_name"])
            
            for inst in inst_list:
                print(f"  {inst['instance_name']:<30} (最佳makespan: {inst['best_makespan']:.2f})")

def save_to_file(label_to_instances, instance_details, output_file="label_distribution_report.txt"):
    """保存报告到文件"""
    import sys
    original_stdout = sys.stdout
    
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_report(label_to_instances, instance_details)
        sys.stdout = original_stdout
    
    print(f"\n报告已保存到: {output_file}")

def save_to_json(label_to_instances, output_file="label_distribution.json"):
    """保存为JSON格式（便于后续程序处理）"""
    result = {}
    
    for method_idx in range(8):
        instances = label_to_instances[method_idx]
        result[METHOD_NAMES[method_idx]] = {
            "label_index": method_idx,
            "count": len(instances),
            "instances": [
                {
                    "instance_name": inst["instance_name"],
                    "dataset": inst["dataset"],
                    "best_makespan": inst["best_makespan"]
                }
                for inst in instances
            ]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"JSON数据已保存到: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析数据标签分布')
    parser.add_argument('--label_root_path', type=str, 
                        default='/workspace/Project_v3/init_validity_result_new',
                        help='标签文件根目录')
    parser.add_argument('--label_name', type=str, default='mean',
                        help='使用的标签名称 (mean, min, max等)')
    parser.add_argument('--output_txt', type=str, default='label_distribution_report.txt',
                        help='文本报告输出文件名')
    parser.add_argument('--output_json', type=str, default='label_distribution.json',
                        help='JSON数据输出文件名')
    parser.add_argument('--no_save', action='store_true',
                        help='不保存文件，仅在终端显示')
    
    args = parser.parse_args()
    
    # 分析标签分布
    print(f"正在分析标签分布...")
    print(f"标签目录: {args.label_root_path}")
    print(f"使用标签: {args.label_name}\n")
    
    label_to_instances, instance_details = analyze_label_distribution(
        args.label_root_path, 
        args.label_name
    )
    
    # 打印报告
    print_report(label_to_instances, instance_details)
    
    # 保存到文件
    if not args.no_save:
        print("\n" + "=" * 80)
        save_to_file(label_to_instances, instance_details, args.output_txt)
        save_to_json(label_to_instances, args.output_json)
        