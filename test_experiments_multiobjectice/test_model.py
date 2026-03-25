import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data as Graph
import argparse
import os
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入项目中的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model


def convert_fjs_to_graph(fjs_path, device=None):
    """
    将单个.fjs文件转换为图数据（无需标签）
    
    Args:
        fjs_path: .fjs文件路径
        device: 设备（cuda或cpu）
    
    Returns:
        PyG Data对象
    """
    with open(fjs_path, "r") as f:
        fjs_lines = f.readlines()

    # 兼容两种格式
    first_line_values = list(fjs_lines[0].split())
    if len(first_line_values) == 3:
        job_num, machine_num, _ = first_line_values
    else:
        job_num, machine_num = first_line_values

    job_num = int(job_num)
    machine_num = int(machine_num)

    # X: [虚拟头节点，虚拟尾节点，机器节点, 作业节点]
    X = [[1, 0, 0, 0], [0, 1, 0, 0]] + [[0, 0, 1, 0] for _ in range(machine_num)]
    edge_index = []
    edge_attr = []

    for job_id in range(job_num):
        data_iter = iter(list(map(int, fjs_lines[job_id + 1].split())))
        operator_num = next(data_iter)
        for operator_id in range(operator_num):
            operator_node_id = len(X)
            X.append([0, 0, 0, 1])
            if operator_id == 0:
                edge_index.append([0, operator_node_id])
            else:
                edge_index.append([operator_node_id - 1, operator_node_id])
            edge_attr.append([1, 0])
            operator_machine_num = next(data_iter)
            for operator_machine_id in range(operator_machine_num):
                machine_id = next(data_iter) - 1
                time = next(data_iter)
                edge_index.append([machine_id + 2, operator_node_id])
                edge_attr.append([0, time])

        edge_index.append([len(X) - 1, 1])
        edge_attr.append([1, 0])

    graph = Graph(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
        edge_attr=torch.tensor(edge_attr, dtype=torch.float))

    if device is not None:
        graph = graph.to(device)
    
    return graph


def load_fjs_files(fjs_dir, device=None):
    """
    从目录加载所有.fjs文件
    
    Args:
        fjs_dir: .fjs文件目录
        device: 设备
    
    Returns:
        graphs: 图数据列表
        filenames: 文件名列表
    """
    graphs = []
    filenames = []
    
    # 遍历目录中的所有.fjs文件
    for filename in sorted(os.listdir(fjs_dir)):
        if filename.endswith('.fjs'):
            fjs_path = os.path.join(fjs_dir, filename)
            try:
                graph = convert_fjs_to_graph(fjs_path, device=device)
                graphs.append(graph)
                filenames.append(filename)
            except Exception as e:
                print(f"⚠ 加载文件失败: {filename}, 错误: {e}")
                continue
    
    return graphs, filenames


class ModelTester:
    """模型测试类 - 仅推理模式（无需标签）"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        
        # 创建测试结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(args.output_dir, f"test_{timestamp}")
        os.makedirs(self.save_path, exist_ok=True)
        
        # 类别名称（与训练时保持一致）
        self.class_names = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 'LWKR_SPT', 'MWKR_SPT']
        
        print(f"="*60)
        print(f"测试配置")
        print(f"="*60)
        print(f"测试结果保存路径: {self.save_path}")
        print(f"FJS文件目录: {args.test_fjs_path}")
        print(f"使用设备: {self.device}")
        
        # 加载测试数据集（只加载.fjs文件，无需标签）
        print(f"\n正在加载测试数据集（仅.fjs文件）...")
        self.graphs, self.filenames = load_fjs_files(
            args.test_fjs_path, 
            device=self.device
        )
        
        if len(self.graphs) == 0:
            raise ValueError(f"未在目录 {args.test_fjs_path} 中找到任何.fjs文件！")
        
        print(f"✓ 成功加载 {len(self.graphs)} 个测试样本")
        
        # 加载模型
        print(f"\n正在加载训练好的模型...")
        print(f"  Checkpoint路径: {args.checkpoint_path}")
        
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {args.checkpoint_path}")
        
        checkpoint = torch.load(args.checkpoint_path, map_location=self.device)
        
        # 从checkpoint中提取配置信息
        if 'epoch' in checkpoint:
            print(f"  训练Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  验证损失: {checkpoint['val_loss']:.4f}")
        
        # 初始化模型（需要与训练时的配置一致）
        self.model = get_model(
            args.model_name,
            node_features=args.node_features,
            edge_features=args.edge_features,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ 模型加载完成")
        print(f"  模型名称: {args.model_name}")
        print(f"="*60 + "\n")
    
    def predict(self):
        """执行推理预测"""
        print("="*60)
        print("开始推理")
        print("="*60 + "\n")
        
        self.model.eval()
        
        all_probs = []
        predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.graphs, desc='推理中')
            for idx, graph in enumerate(pbar):
                # 前向传播
                # batch参数需要为每个图创建
                batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                output = self.model(graph.x, graph.edge_index, graph.edge_attr, batch)
                
                # 计算概率
                probs = torch.exp(output).cpu().numpy()[0]  # log_softmax -> softmax
                all_probs.append(probs)
                
                # 预测标签
                pred_label = output.argmax(dim=1).cpu().numpy()[0]
                
                # 保存详细预测信息
                predictions.append({
                    'sample_index': idx,
                    'instance_name': self.filenames[idx],
                    'predicted_label': int(pred_label),
                    'predicted_label_name': self.class_names[pred_label],
                    'prediction_probs': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.class_names, probs)
                    },
                    'num_nodes': int(graph.x.shape[0]),
                    'num_edges': int(graph.edge_index.shape[1]),
                })
                
                pbar.set_postfix({
                    'pred': self.class_names[pred_label],
                    'conf': f'{probs[pred_label]:.3f}'
                })
        
        all_probs = np.array(all_probs)
        
        print("\n" + "="*60)
        print("推理完成")
        print("="*60)
        print(f"推理样本数: {len(all_probs)}")
        print(f"\n预测类别分布:")
        pred_labels = [p['predicted_label'] for p in predictions]
        for i, class_name in enumerate(self.class_names):
            count = pred_labels.count(i)
            print(f"  {class_name}: {count} ({100*count/len(pred_labels):.1f}%)")
        print("="*60 + "\n")
        
        # 保存详细结果
        results = {
            'test_info': {
                'checkpoint_path': self.args.checkpoint_path,
                'test_fjs_path': self.args.test_fjs_path,
                'num_samples': len(all_probs),
                'model_name': self.args.model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'predictions': predictions
        }
        
        # 保存结果到JSON
        results_path = os.path.join(self.save_path, 'predictions.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✓ 预测结果已保存至: {results_path}")
        
        # 【关键】生成概率分布
        self.save_probability_distribution(all_probs, predictions)
        
        # 绘制预测分布统计
        self.plot_prediction_statistics(predictions, all_probs)
        
        print("\n" + "="*60)
        print("测试完成！")
        print(f"所有结果已保存至: {self.save_path}")
        print("="*60 + "\n")
        
        return results
    
    def save_probability_distribution(self, all_probs, predictions):
        """
        保存并可视化概率分布（用于下游任务）
        这是最关键的输出
        """
        results = {}
        
        # 方案1: 简单平均分布（最常用）
        avg_distribution = np.mean(all_probs, axis=0)
        avg_distribution = avg_distribution / np.sum(avg_distribution)
        results['average_distribution'] = avg_distribution.tolist()
        
        # 方案2: 中位数分布（更鲁棒，不受异常值影响）
        median_distribution = np.median(all_probs, axis=0)
        median_distribution = median_distribution / np.sum(median_distribution)
        results['median_distribution'] = median_distribution.tolist()
        
        # 方案3: 加权平均分布（根据预测置信度加权）
        confidences = np.max(all_probs, axis=1)  # 每个样本的最大概率作为置信度
        weights = np.exp(confidences * 5)  # 放大差异
        weights = weights / np.sum(weights)
        weighted_distribution = np.sum(all_probs * weights[:, np.newaxis], axis=0)
        weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
        results['weighted_distribution'] = weighted_distribution.tolist()
        results['weights_stats'] = {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'max': float(np.max(weights)),
            'min': float(np.min(weights))
        }
        
        # 方案4: 投票分布（基于预测标签的频率）
        pred_labels = [p['predicted_label'] for p in predictions]
        vote_distribution = np.zeros(self.args.num_classes)
        for label in pred_labels:
            vote_distribution[label] += 1
        vote_distribution = vote_distribution / np.sum(vote_distribution)
        results['vote_distribution'] = vote_distribution.tolist()
        
        # 添加元数据
        results['class_names'] = self.class_names
        results['num_classes'] = self.args.num_classes
        results['num_samples'] = len(all_probs)
        
        # 计算每个分布的统计信息
        results['distribution_statistics'] = {}
        for key in ['average_distribution', 'median_distribution', 'weighted_distribution', 'vote_distribution']:
            if key in results:
                dist = np.array(results[key])
                # 熵（越高表示分布越均匀）
                entropy = -np.sum(dist * np.log(dist + 1e-10))
                # 基尼系数（越低表示分布越均匀）
                gini = 1 - np.sum(dist ** 2)
                # 最大概率
                max_prob = np.max(dist)
                # 最大概率对应的类别
                max_class = self.class_names[np.argmax(dist)]
                
                results['distribution_statistics'][key] = {
                    'entropy': float(entropy),
                    'gini_coefficient': float(gini),
                    'max_probability': float(max_prob),
                    'dominant_class': max_class
                }
        
        # 保存到JSON
        output_path = os.path.join(self.save_path, 'probability_distribution.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ 概率分布已保存至: {output_path}")
        
        # 打印分布信息
        print(f"\n" + "="*60)
        print("推荐使用的概率分布（用于下游任务权重分配）")
        print("="*60)
        
        for method_name in ['average_distribution', 'median_distribution', 'weighted_distribution', 'vote_distribution']:
            if method_name in results:
                dist = results[method_name]
                stats = results['distribution_statistics'][method_name]
                print(f"\n方法: {method_name}")
                for class_name, prob in zip(self.class_names, dist):
                    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
                print(f"  统计信息:")
                print(f"    - 熵值: {stats['entropy']:.4f}")
                print(f"    - 基尼系数: {stats['gini_coefficient']:.4f}")
                print(f"    - 主导类别: {stats['dominant_class']} ({stats['max_probability']*100:.2f}%)")
        
        print(f"\n" + "="*60)
        print("建议:")
        print("  - 一般情况: 使用 'average_distribution'")
        print("  - 有异常值: 使用 'median_distribution'")
        print("  - 重视置信度: 使用 'weighted_distribution'")
        print("  - 简单投票: 使用 'vote_distribution'")
        print("="*60)
        
        # 绘制可视化图表
        self.plot_probability_distribution_comparison(results)
        
        return results
    
    def plot_probability_distribution_comparison(self, results):
        """
        绘制概率分布对比图
        """
        methods = []
        distributions = []
        
        method_labels = {
            'average_distribution': 'Average Distribution',
            'median_distribution': 'Median Distribution',
            'weighted_distribution': 'Weighted Distribution',
            'vote_distribution': 'Vote Distribution'
        }
        
        for method_key, label in method_labels.items():
            if method_key in results:
                methods.append(label)
                distributions.append(results[method_key])
        
        if len(methods) == 0:
            return
        
        # 创建对比柱状图（2x2布局）
        n_methods = len(methods)
        n_cols = 2
        n_rows = (n_methods + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.args.num_classes))
        
        for idx, (method, dist) in enumerate(zip(methods, distributions)):
            ax = axes[idx]
            bars = ax.bar(range(self.args.num_classes), dist, color=colors, 
                         edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(self.args.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(method, fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(max(d) for d in distributions) * 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # 标注概率值
            for bar, prob in zip(bars, dist):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余的子图
        for idx in range(len(methods), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Predicted Probability Distribution Comparison\n(For Downstream Task Weights)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_path, 'probability_distribution_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 概率分布对比图已保存至: {save_path}")
    
    def plot_prediction_statistics(self, predictions, all_probs):
        """绘制预测统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 预测类别分布（饼图）
        ax = axes[0, 0]
        pred_labels = [p['predicted_label'] for p in predictions]
        label_counts = [pred_labels.count(i) for i in range(self.args.num_classes)]
        colors = plt.cm.Set3(np.linspace(0, 1, self.args.num_classes))
        ax.pie(label_counts, labels=self.class_names, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        
        # 2. 预测置信度分布（直方图）
        ax = axes[0, 1]
        confidences = [max(p['prediction_probs'].values()) for p in predictions]
        ax.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 每个类别的平均概率（柱状图）
        ax = axes[1, 0]
        avg_probs = np.mean(all_probs, axis=0)
        bars = ax.bar(range(self.args.num_classes), avg_probs, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(self.args.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_ylabel('Average Probability', fontsize=12)
        ax.set_title('Average Probability per Class', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, prob in zip(bars, avg_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # 4. 样本规模分布（节点数）
        ax = axes[1, 1]
        node_counts = [p['num_nodes'] for p in predictions]
        ax.hist(node_counts, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Instance Size Distribution', fontsize=14, fontweight='bold')
        ax.axvline(np.mean(node_counts), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(node_counts):.1f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_path, 'prediction_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 预测统计图已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='GNN模型推理脚本 - FJSP问题（仅推理，无需标签）')
    
    # 测试数据相关参数
    parser.add_argument('--test_fjs_path', type=str, 
                       default='./test_experiments/test_dataset',
                       help='测试集FJS文件目录路径')
    
    # 模型相关参数
    parser.add_argument('--checkpoint_path', type=str,
                       default='./checkpoints/NNConv_Deep_Attention_Pooling/20251204_092448/checkpoint_best.pt',
                       help='训练好的模型checkpoint路径')
    parser.add_argument('--model_name', type=str, 
                       default='NNConv_Deep_Attention_Pooling',
                       help='模型名称（需与checkpoint一致）')
    
    # 模型配置参数（需与训练时一致）
    parser.add_argument('--node_features', type=int, default=4,
                       help='节点特征维度')
    parser.add_argument('--edge_features', type=int, default=2,
                       help='边特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='隐藏层维度')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='分类类别数')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./test_experiments/results',
                       help='测试结果保存目录')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用GPU')
    
    args = parser.parse_args()
    
    # 创建测试器并执行推理
    tester = ModelTester(args)
    tester.predict()


if __name__ == '__main__':
    main()
    