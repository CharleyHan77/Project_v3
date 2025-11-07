import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def load_training_history(checkpoint_dir):
    """加载训练历史"""
    history_path = os.path.join(checkpoint_dir, 'train_history.json')
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if not os.path.exists(history_path):
        return None, None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return history, config

def find_model_runs(base_dir, model_names=None):
    """
    查找所有模型的训练记录
    
    Args:
        base_dir: checkpoints根目录
        model_names: 要对比的模型名称列表，None表示所有模型
    
    Returns:
        {model_name: [(timestamp, history, config), ...]}
    """
    base_path = Path(base_dir)
    results = {}
    
    if not base_path.exists():
        print(f"目录不存在: {base_dir}")
        return results
    
    # 遍历每个模型目录
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 如果指定了模型列表，只处理列表中的模型
        if model_names and model_name not in model_names:
            continue
        
        # 查找该模型的所有训练记录
        model_runs = []
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            timestamp = run_dir.name
            history, config = load_training_history(str(run_dir))
            
            if history:
                model_runs.append((timestamp, history, config))
        
        if model_runs:
            # 按时间戳排序，取最新的
            model_runs.sort(key=lambda x: x[0], reverse=True)
            results[model_name] = model_runs
    
    return results

def plot_comparison(results, metric_key, metric_name, save_dir, use_latest_only=True):
    """
    绘制多个模型的指标对比图
    
    Args:
        results: find_model_runs返回的结果
        metric_key: 指标在history中的键名
        metric_name: 指标显示名称
        save_dir: 保存目录
        use_latest_only: 是否只使用每个模型的最新训练记录
    """
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    
    for model_name, runs in results.items():
        # 选择要绘制的训练记录
        selected_runs = [runs[0]] if use_latest_only else runs
        
        for run_idx, (timestamp, history, config) in enumerate(selected_runs):
            if metric_key not in history:
                continue
            
            values = history[metric_key]
            epochs = range(1, len(values) + 1)
            
            # 构建标签
            if use_latest_only:
                label = model_name
            else:
                label = f"{model_name}_{timestamp}"
            
            # 如果有config，添加关键参数信息
            if config:
                label += f" (lr={config.get('lr', 'N/A')}, hd={config.get('hidden_dim', 'N/A')})"
            
            # 绘制曲线
            plt.plot(epochs, values, label=label, 
                    linewidth=2.5, marker='o', markersize=4,
                    color=colors[color_idx % len(colors)], alpha=0.8)
            
            color_idx += 1
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.title(f'{metric_name} Comparison Across Models', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'comparison_{metric_key}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {metric_name}对比图已保存: {save_path}")
    plt.close()

def plot_all_metrics_comparison(results, save_dir, use_latest_only=True):
    """绘制所有关键指标的对比图"""
    metrics = [
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy (%)'),
        ('val_macro_f1', 'Macro F1-Score (%)'),
        ('val_weighted_f1', 'Weighted F1-Score (%)'),
        ('val_roc_auc', 'ROC-AUC (%)'),
        ('train_loss', 'Training Loss'),
    ]
    
    for metric_key, metric_name in metrics:
        plot_comparison(results, metric_key, metric_name, save_dir, use_latest_only)

def generate_summary_table(results, save_dir):
    """生成模型对比汇总表"""
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'model_comparison_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("模型对比汇总表\n")
        f.write("="*100 + "\n\n")
        
        # 表头
        header = f"{'模型名称':<30} {'时间戳':<18} {'最佳验证损失':<15} {'最高准确率(%)':<15} {'最高F1(%)':<12}\n"
        f.write(header)
        f.write("-"*100 + "\n")
        
        # 每个模型的数据
        for model_name, runs in results.items():
            timestamp, history, config = runs[0]  # 使用最新的训练记录
            
            best_val_loss = min(history.get('val_loss', [float('inf')]))
            best_accuracy = max(history.get('val_accuracy', [0]))
            best_f1 = max(history.get('val_macro_f1', [0]))
            
            row = f"{model_name:<30} {timestamp:<18} {best_val_loss:<15.4f} {best_accuracy:<15.2f} {best_f1:<12.2f}\n"
            f.write(row)
        
        f.write("="*100 + "\n")
    
    print(f"✓ 汇总表已保存: {summary_path}")
    
    # 同时打印到控制台
    with open(summary_path, 'r', encoding='utf-8') as f:
        print("\n" + f.read())

def main():
    parser = argparse.ArgumentParser(description='对比多个模型的训练结果')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='checkpoints根目录 (默认: ./checkpoints)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='要对比的模型名称列表，不指定则对比所有模型')
    parser.add_argument('--save_dir', type=str, default='./model_comparison',
                        help='对比图表保存目录 (默认: ./model_comparison)')
    parser.add_argument('--use_all_runs', action='store_true',
                        help='是否绘制每个模型的所有训练记录（默认只使用最新的）')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='不使用时间戳子目录（直接保存在save_dir根目录）')
    
    args = parser.parse_args()
    
    # 🔧 新增：创建带时间戳的子目录
    if not args.no_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_save_dir = os.path.join(args.save_dir, timestamp)
    else:
        actual_save_dir = args.save_dir
    
    print("\n" + "="*80)
    print("模型对比分析工具")
    print("="*80 + "\n")
    
    # 查找所有训练记录
    print(f"正在扫描目录: {args.checkpoint_dir}")
    results = find_model_runs(args.checkpoint_dir, args.models)
    
    if not results:
        print("❌ 未找到任何训练记录！")
        return
    
    print(f"✓ 找到 {len(results)} 个模型的训练记录:\n")
    for model_name, runs in results.items():
        print(f"  - {model_name}: {len(runs)} 次训练")
    
    # 显示保存路径
    print(f"\n📁 对比结果将保存至: {actual_save_dir}")
    
    # 生成汇总表
    print("\n正在生成汇总表...")
    generate_summary_table(results, actual_save_dir)
    
    # 绘制对比图
    print("\n正在生成对比图表...")
    use_latest_only = not args.use_all_runs
    plot_all_metrics_comparison(results, actual_save_dir, use_latest_only)
    
    print("\n" + "="*80)
    print("✓ 对比分析完成！")
    print(f"所有结果已保存至: {actual_save_dir}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
    