import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import Subset
import argparse
import os
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib
from scipy.stats import kendalltau
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import wandb

from model import get_model, MODEL_REGISTRY # 支持模型注册选择
from dataset_nsga2 import Dataset

####################################################多目标
"""
训练说明：
- 任务类型：分类任务（选择最佳的初始化方法）
- 标签格式：
- 标签转换：将性能值转为类别（argmin，因为性能值越小越好）
- 损失函数：
- 评估指标：分类准确率
"""

# class FocalLoss(nn.Module):
#     """
#     Focal Loss: 专门设计用于处理类别极度不平衡的问题
#     论文: https://arxiv.org/abs/1708.02002
#     """
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha  # 类别权重
#         self.gamma = gamma  # 聚焦参数（2-5之间，越大越关注难分类样本）
#         self.reduction = reduction
    
#     def forward(self, log_probs, targets):
#         """
#         Args:
#             log_probs: [batch_size, num_classes] log概率（log_softmax输出）
#             targets: [batch_size] 类别索引
#         """
#         # 转换为概率
#         probs = torch.exp(log_probs)
        
#         # 获取目标类别的概率和log概率
#         targets = targets.view(-1)
#         pt = probs.gather(1, targets.view(-1, 1)).view(-1)  # 正确类别的概率
#         log_pt = log_probs.gather(1, targets.view(-1, 1)).view(-1)
        
#         # Focal weight: (1-pt)^gamma
#         # 难分类样本的pt小，focal_weight大，权重高
#         focal_weight = (1 - pt) ** self.gamma
        
#         # 应用类别权重
#         if self.alpha is not None:
#             alpha_t = self.alpha.gather(0, targets)
#             focal_weight = alpha_t * focal_weight
        
#         loss = -focal_weight * log_pt
        
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        
        # 设置随机种子
        self.set_seed(args.seed)
        
        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(args.save_dir, args.model_name, timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        # 初始化 wandb
        wandb.init(
            project="Project_v3.1_multiobjective",  # 项目名称，可以自定义
            name=f"{args.model_name}_{timestamp}",  # 运行名称
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "accumulation_steps": args.accumulation_steps,
                "train_ratio": args.train_ratio,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "seed": args.seed,
                "node_features": args.node_features,
                "edge_features": args.edge_features,
                "num_classes": args.num_classes,
            },
            dir=self.save_path  # wandb 日志保存到相同目录
        )

        # if args.num_classes == 8:
        #     self.class_names = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 
        #                        'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
        # elif args.num_classes == 3:
        #     self.class_names = ['heuristic', 'mixed', 'random']
        # else:
        #     # 对于其他类别数，使用通用名称
        #     self.class_names = [f'class_{i}' for i in range(args.num_classes)]
        self.class_names = [
            'FIFO_SPT', 
            #'FIFO_EET', 
            'MOPNR_SPT', 
            'MOPNR_EET', 
            #'LWKR_SPT', 
            #'LWKR_EET',
            'MWKR_SPT',
            'MWKR_EET'
            ]
        # 与 multiobjective 不同的5个推荐规则
        
        # 加载数据集
        print(f"正在加载数据集...")
        full_dataset = Dataset(args.fjs_root_path, args.label_root_path, device="cuda")
        
        # 分层采样验证集
        # 先提取所有样本的标签
        all_labels = [data.y.argmin().item() for data in full_dataset]
        
        # 使用sklearn的train_test_split进行分层采样
        indices = list(range(len(full_dataset)))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=1 - args.train_ratio,
            stratify=all_labels,  # 关键参数：按标签分层
            random_state=args.seed
        )
        
        # 使用Subset创建训练集和验证集
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # 记录验证集样本的详细信息
        val_samples_info = []
        for idx in val_indices:
            sample = full_dataset[idx]
            val_samples_info.append({
                'dataset_index': int(idx),
                'instance_name': full_dataset.filenames[idx],
                'true_label': int(sample.y.argmin().item()),
                'num_nodes': int(sample.x.shape[0]),
                'num_edges': int(sample.edge_index.shape[1])
            })
        
        val_samples_path = os.path.join(self.save_path, 'validation_samples.json')
        with open(val_samples_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_samples': len(val_indices),
                'samples': val_samples_info,
                'train_indices': [int(i) for i in train_indices],
                'val_indices': [int(i) for i in val_indices],
            }, f, indent=4, ensure_ascii=False)
        print(f"验证集样本信息已保存至: {val_samples_path}")

        wandb.config.update({
            "num_train_samples": len(train_indices),
            "num_val_samples": len(val_indices),
        })
        val_table = wandb.Table(columns=["Dataset_Index", "True_Label", "Label_Name", "Num_Nodes", "Num_Edges"])
        for info in val_samples_info:
            val_table.add_data(
                info['dataset_index'],
                info['true_label'],
                self.class_names[info['true_label']],
                info['num_nodes'],
                info['num_edges']
            )
        wandb.log({"validation_samples": val_table})
        artifact = wandb.Artifact('validation_samples', type='dataset')
        artifact.add_file(val_samples_path)
        wandb.log_artifact(artifact)

        #####################################################        
        # 计算类别权重以处理类别不平衡问题
        print(f"\n正在计算类别权重...")
        train_labels = [full_dataset[i].y.argmin().item() for i in train_indices]
        
        # 统计每个类别的样本数
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        print(f"训练集各类别样本数: {dict(zip(unique_classes, class_counts))}")
        
        # 计算反比例权重：样本越少的类别权重越高
        total_samples = len(train_labels)
        class_weights = np.zeros(args.num_classes)
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = total_samples / (args.num_classes * count)
        
        # 如果某个类别没有样本，权重设为0
        class_weights[class_weights == np.inf] = 0
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f"类别权重: {class_weights}")
        #####################################################
        # 统计训练集和验证集的类别分布
        train_methods = [full_dataset[i].y.argmin().item() for i in train_indices]
        val_methods = [full_dataset[i].y.argmin().item() for i in val_indices]
        
        train_nodes = [data.x.shape[0] for data in self.train_dataset]
        val_nodes = [data.x.shape[0] for data in self.val_dataset]
        
        print(f"\n数据集划分统计:")
        print(f"  总样本数: {len(full_dataset)}")
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        
        print(f"\n训练集类别分布:")
        for method_id, method_name in zip(range(self.args.num_classes), self.class_names):
            count = train_methods.count(method_id)
            print(f"  {method_name}: {count} ({100*count/len(train_methods):.1f}%)")
        
        print(f"\n验证集类别分布:")
        for method_id, method_name in zip(range(self.args.num_classes), self.class_names):
            count = val_methods.count(method_id)
            print(f"  {method_name}: {count} ({100*count/len(val_methods):.1f}%)")
        
        print(f"\n训练集节点数统计: min={min(train_nodes)}, max={max(train_nodes)}, mean={np.mean(train_nodes):.1f}")
        print(f"验证集节点数统计: min={min(val_nodes)}, max={max(val_nodes)}, mean={np.mean(val_nodes):.1f}")
        
        print(f"\n训练模式: 单图训练 + 梯度累积({args.accumulation_steps}步)")
        print(f"等效批次大小: {args.accumulation_steps}")

        # # 使用Focal Loss替代普通NLLLoss
        # self.criterion = FocalLoss(
        #     alpha=self.class_weights,
        #     gamma=2.0  # 增大gamma更关注难样本，8分类建议3-4
        # )
        # print(f"使用 Focal Loss (gamma=2.0) 处理类别不平衡")

        
        # 创建数据加载器 - 由于图大小不一致，每次加载一个图
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=1,  # 每次处理一个图
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # 每次处理一个图
            shuffle=False
        )
        
        # 初始化模型 - 工厂函数选择模型
        self.model = get_model(
            args.model_name,
            node_features=args.node_features,
            edge_features=args.edge_features,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes
        ).to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience,
            verbose=True
        )
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_weighted_f1': [],
            'val_roc_auc': [],
            'lr': [],
            'train_loss_std': [],  # 每个epoch内batch损失的标准差
            'train_loss_min': [],  # 每个epoch内batch损失的最小值
            'train_loss_max': [],  # 每个epoch内batch损失的最大值
            'val_loss_std': [],    # 验证集每个batch损失的标准差
            'val_loss_min': [],    # 验证集每个batch损失的最小值
            'val_loss_max': [],    # 验证集每个batch损失的最大值
            # 【新增】分布匹配指标
            'val_js_divergence': [],
            'val_cosine_similarity': [],
            'val_top3_distribution_match': [],
            'val_brier_score': [],
            'val_top3_accuracy': [],
            'val_adaptive_topk_distribution_match': [],  # 动态top-k
            'val_adaptive_topk_accuracy': [],  # 动态top-k
            'val_avg_k_value': [],  # 平均k值
        }

        # 添加梯度和激活监测
        self.train_history['grad_norm'] = []           # 每个epoch的平均梯度范数
        self.train_history['grad_norm_std'] = []       # 梯度范数标准差
        self.train_history['layer_activations'] = []   # 中间层激活统计
        
        self.best_val_loss = float('inf')
        
        # 保存配置
        self.save_config()
        print(f"使用模型: {args.model_name}")
        
        print(f"模型已初始化，使用设备: {self.device}")
        # print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def set_seed(self, seed):
        """设置随机种子以确保可重复性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def compute_grad_norm(self):
        """计算所有参数的梯度范数"""
        total_norm = 0.0
        grad_norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        return total_norm, grad_norms

    def save_best_probability_distribution(self, all_labels, all_probs, dist_metrics_list, epoch='final'):
        """
        保存用于下游任务的最优概率分布向量
        
        Args:
            all_labels: 真实标签 [n_samples]
            all_probs: 所有样本的预测概率 [n_samples, num_classes]
            dist_metrics_list: 每个样本的分布匹配指标列表
            epoch: 当前epoch或'final'
        
        Returns:
            dict: 包含最优分布及相关信息
        """
        results = {}
        
        # 方案1: 简单平均分布（最稳定）
        avg_distribution = np.mean(all_probs, axis=0)
        avg_distribution = avg_distribution / np.sum(avg_distribution)  # 归一化确保和为1
        results['average_distribution'] = avg_distribution.tolist()
        
        # 方案2: 选择分布匹配最好的单个样本
        # 这里使用余弦相似度作为标准（值越大越好）
        if len(dist_metrics_list) > 0:
            cosine_similarities = [m['cosine_similarity'] for m in dist_metrics_list]
            best_idx = np.argmax(cosine_similarities)
            best_sample_distribution = all_probs[best_idx]
            best_sample_distribution = best_sample_distribution / np.sum(best_sample_distribution)
            results['best_sample_distribution'] = best_sample_distribution.tolist()
            results['best_sample_metrics'] = dist_metrics_list[best_idx]
            results['best_sample_index'] = int(best_idx)
        
        # 方案3: 加权平均分布（根据分布匹配质量加权）
        if len(dist_metrics_list) > 0:
            cosine_similarities = np.array([m['cosine_similarity'] for m in dist_metrics_list])
            # 使用softmax将相似度转换为权重
            weights = np.exp(cosine_similarities * 5)  # 放大差异
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
        
        # 添加元数据
        results['class_names'] = self.class_names
        results['num_classes'] = self.args.num_classes
        results['num_samples'] = len(all_probs)
        
        # 计算每个方案的熵（熵越高表示分布越均匀）
        for key in ['average_distribution', 'best_sample_distribution', 'weighted_distribution']:
            if key in results:
                dist = np.array(results[key])
                entropy = -np.sum(dist * np.log(dist + 1e-10))
                results[f'{key}_entropy'] = float(entropy)
        
        # 保存到文件
        if epoch == 'final':
            output_path = os.path.join(self.save_path, 'best_probability_distribution.json')
        else:
            output_path = os.path.join(self.save_path, f'probability_distribution_epoch{epoch}.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n  ✓ 最优概率分布已保存至: {output_path}")
        
        # 打印分布信息
        print(f"\n  推荐使用的概率分布（用于下游任务权重分配）:")
        print(f"  {'='*60}")
        
        for method_name in ['average_distribution', 'weighted_distribution', 'best_sample_distribution']:
            if method_name in results:
                dist = results[method_name]
                print(f"\n  方法: {method_name}")
                for i, (class_name, prob) in enumerate(zip(self.class_names, dist)):
                    print(f"    {class_name}: {prob:.4f} ({prob*100:.2f}%)")
                if f'{method_name}_entropy' in results:
                    print(f"    熵值: {results[f'{method_name}_entropy']:.4f}")
        
        print(f"\n  {'='*60}")
        print(f"  建议: 一般使用 'weighted_distribution' 作为下游任务权重")
        
        # 绘制可视化图表
        self.plot_probability_distribution_comparison(results, epoch)
        
        # 记录到wandb
        self.log_best_distribution_to_wandb(results, epoch)
        
        return results

    def js_divergence(self, p, q):
        """
        计算JS散度（对称版本的KL散度）
        p, q: [batch_size, num_classes] 概率分布（已softmax）
        返回：JS散度值（越小越好，范围0-1）
        """
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(torch.log(p + 1e-10), m, reduction='batchmean')
        kl_qm = F.kl_div(torch.log(q + 1e-10), m, reduction='batchmean')
        return 0.5 * (kl_pm + kl_qm)

    def adaptive_top_k_hybrid(self, probs, gap_threshold=0.15, cumulative_threshold=0.9, 
                          min_k=1, max_k=5):
        """
        混合自适应k选择策略
        
        同时考虑：
        1. 概率差距：相邻排名差距小于阈值时继续
        2. 累积概率：累积概率未达到阈值时继续
        3. 最小概率：单个类别概率大于阈值时继续
        
        Args:
            probs: [batch_size, num_classes] 概率分布
            gap_threshold: 概率差距阈值（默认0.15）
            cumulative_threshold: 累积概率阈值（默认0.9）
            min_k: 最小k值（默认1）
            max_k: 最大k值（默认5）
        
        Returns:
            k值列表（每个样本一个k）
        """
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        k_values = []
        
        for i in range(probs.size(0)):
            k = min_k
            cumsum = sorted_probs[i, 0].item()
            
            for j in range(1, min(probs.size(1), max_k)):
                # 条件1：累积概率是否已经足够高
                if cumsum >= cumulative_threshold:
                    break
                
                # 条件2：概率差距是否过大（说明后续类别不重要）
                gap = sorted_probs[i, j-1] - sorted_probs[i, j]
                if gap >= gap_threshold:
                    break
                
                # 条件3：当前类别概率是否太小（不值得考虑）
                if sorted_probs[i, j] < 0.05:  # 小于5%的概率忽略
                    break
                
                k = j + 1
                cumsum += sorted_probs[i, j].item()
            
            k_values.append(k)
        
        return k_values

    def compute_distribution_metrics(self, pred_probs, true_probs):
        """
        计算一批样本的分布匹配指标
        pred_probs: 预测的概率分布 [batch_size, num_classes]
        true_probs: 真实的概率分布 [batch_size, num_classes]
        
        返回：包含所有分布匹配指标的字典
        """
        metrics = {}
        
        # 1. JS散度
        js_div = self.js_divergence(pred_probs, true_probs)
        metrics['js_divergence'] = js_div.item()
        
        # 2. 余弦相似度
        cos_sim = F.cosine_similarity(pred_probs, true_probs, dim=1).mean()
        metrics['cosine_similarity'] = cos_sim.item()
        
        # 3. Top-3分布匹配率
        _, pred_top3 = pred_probs.topk(3, dim=1)
        _, true_top3 = true_probs.topk(3, dim=1)
        
        top3_matches = []
        for i in range(pred_probs.size(0)):
            pred_set = set(pred_top3[i].cpu().numpy())
            true_set = set(true_top3[i].cpu().numpy())
            overlap = len(pred_set & true_set) / 3.0
            top3_matches.append(overlap)
        metrics['top3_distribution_match'] = np.mean(top3_matches)

        
        # 4. Brier Score（均方误差）
        brier = torch.mean((pred_probs - true_probs) ** 2)
        metrics['brier_score'] = brier.item()
        
        # 5. Top-3准确率
        true_label = true_probs.argmax(dim=1)
        _, topk_indices = pred_probs.topk(3, dim=1)
        correct = topk_indices.eq(true_label.view(-1, 1).expand_as(topk_indices))
        top3_acc = correct.any(dim=1).float().mean()
        metrics['top3_accuracy'] = top3_acc.item()

        # 6. 【改进】动态Top-k分布匹配率
        # 分别计算预测和真实概率的自适应k
        pred_k_values = self.adaptive_top_k_hybrid(pred_probs, 
                                                    gap_threshold=0.15, 
                                                    cumulative_threshold=0.85)
        true_k_values = self.adaptive_top_k_hybrid(true_probs, 
                                                    gap_threshold=0.15, 
                                                    cumulative_threshold=0.85)
        
        adaptive_matches = []
        k_distribution = []  # 记录k值分布，用于分析
        
        for i in range(pred_probs.size(0)):
            # 使用两者中较大的k（更宽松的匹配）
            k = max(pred_k_values[i], true_k_values[i])
            k_distribution.append(k)
            
            # 获取top-k索引
            _, pred_topk = pred_probs[i].topk(k)
            _, true_topk = true_probs[i].topk(k)
            
            # 计算重叠率
            pred_set = set(pred_topk.cpu().numpy())
            true_set = set(true_topk.cpu().numpy())
            overlap = len(pred_set & true_set) / k
            adaptive_matches.append(overlap)
        
        metrics['adaptive_topk_distribution_match'] = np.mean(adaptive_matches)
        metrics['avg_k_value'] = np.mean(k_distribution)  # 平均k值，用于监控
        metrics['k_distribution'] = k_distribution  # 完整的k值分布

        # 6.1 【改进】动态Top-k准确率
        true_label = true_probs.argmax(dim=1)
        adaptive_correct = []
        
        for i in range(pred_probs.size(0)):
            k = pred_k_values[i]
            _, topk_indices = pred_probs[i].topk(k)
            correct = true_label[i] in topk_indices
            adaptive_correct.append(correct)
        
        metrics['adaptive_topk_accuracy'] = np.mean(adaptive_correct)
        
        return metrics
    
    def get_layer_activations(self, data):
        """获取中间层的激活统计信息"""
        self.model.eval()
        activations = {}
        
        # 钩子函数来捕获激活
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'zero_ratio': (output.abs() < 1e-6).float().mean().item()  # 接近0的比例
                    }
            return hook
        
        # 注册钩子
        hooks = []
        if hasattr(self.model, 'conv1'):
            hooks.append(self.model.conv1.register_forward_hook(hook_fn('conv1')))
        if hasattr(self.model, 'conv2'):
            hooks.append(self.model.conv2.register_forward_hook(hook_fn('conv2')))
        if hasattr(self.model, 'fc1'):
            hooks.append(self.model.fc1.register_forward_hook(hook_fn('fc1')))
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        self.model.train()
        return activations
    
    def save_config(self):
        """保存训练配置"""
        config = vars(self.args)
        config_path = os.path.join(self.save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"配置已保存至: {config_path}")

    def plot_probability_distribution_comparison(self, results, epoch='final'):
        """
        绘制不同方法得到的概率分布对比图
        """
        methods = []
        distributions = []
        
        method_labels = {
            'average_distribution': 'Average Distribution',
            'weighted_distribution': 'Weighted Average Distribution',
            'best_sample_distribution': 'Best Sample Distribution'
        }
        
        for method_key, label in method_labels.items():
            if method_key in results:
                methods.append(label)
                distributions.append(results[method_key])
        
        if len(methods) == 0:
            return
        
        # 创建对比柱状图
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.args.num_classes))
        
        for idx, (method, dist) in enumerate(zip(methods, distributions)):
            ax = axes[idx]
            bars = ax.bar(range(self.args.num_classes), dist, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(self.args.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(method, fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(max(d) for d in distributions) * 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # 在柱子上标注概率值
            for bar, prob in zip(bars, dist):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Predicted Probability Distribution Comparison (for Downstream Task Weights)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if epoch == 'final':
            save_path = os.path.join(self.save_path, 'best_probability_distribution_comparison.png')
        else:
            save_path = os.path.join(self.save_path, f'probability_distribution_comparison_epoch{epoch}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 概率分布对比图已保存至: {save_path}")

    def log_best_distribution_to_wandb(self, results, epoch='final'):
        """
        将最优概率分布记录到wandb
        """
        prefix = "final/" if epoch == 'final' else f"epoch_{epoch}/"
        
        # 记录推荐的加权分布
        if 'weighted_distribution' in results:
            for class_name, prob in zip(self.class_names, results['weighted_distribution']):
                wandb.log({
                    f"{prefix}best_distribution/{class_name}": prob
                })
            
            wandb.log({
                f"{prefix}best_distribution/entropy": results.get('weighted_distribution_entropy', 0)
            })
        
        # 创建表格对比
        if epoch == 'final':
            table_data = []
            for i, class_name in enumerate(self.class_names):
                row = [class_name]
                for method in ['average_distribution', 'weighted_distribution', 'best_sample_distribution']:
                    if method in results:
                        row.append(results[method][i])
                table_data.append(row)
            
            columns = ['类别']
            if 'average_distribution' in results:
                columns.append('平均分布')
            if 'weighted_distribution' in results:
                columns.append('加权分布')
            if 'best_sample_distribution' in results:
                columns.append('最佳样本')
            
            wandb.log({
                f"{prefix}best_distribution/comparison_table": wandb.Table(
                    columns=columns,
                    data=table_data
                )
            })
        
        print(f"  ✓ 最优分布已记录到wandb")
    
    def train_epoch(self, epoch):
        """训练一个epoch - 每次处理一个图"""
        self.model.train()
        total_loss = 0
        accumulation_steps = self.args.accumulation_steps

        # 新增：记录每个batch的损失
        batch_losses = []
        batch_grad_norms = []

        # 熵正则化系数（可通过args传入）
        entropy_weight = getattr(self.args, 'entropy_weight', 0.05)  # 默认0.01
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [训练]')
        for batch_idx, data in enumerate(pbar):
            # 前向传播
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

            ############################ 分类/回归 标签转换 ############################

            temperature = 0.02  # 温度参数（使得分布更平滑）
            class_label = F.softmax(-data.y / temperature, dim=0).unsqueeze(0)
            loss = F.kl_div(output, class_label, reduction='batchmean')

            ############################ 分类/回归 标签转换 ############################

            # 记录原始损失值（在梯度累积之前）
            batch_losses.append(loss.item())
            
            # 梯度累积
            loss = loss / accumulation_steps
            # 反向传播
            loss.backward()

            # 【新增】在backward之后计算梯度范数
            grad_norm, _ = self.compute_grad_norm()
            batch_grad_norms.append(grad_norm)
            
            # 每accumulation_steps步更新一次权重
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            
            # 更新进度条
            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'grad_norm': f'{grad_norm:.4f}'  # 显示梯度范数
                })
        
        # 处理最后剩余的梯度
        if len(self.train_loader) % accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)

        # 新增：计算batch损失的统计信息
        batch_losses_array = np.array(batch_losses)
        batch_grad_norms_array = np.array(batch_grad_norms)
        loss_std = np.std(batch_losses_array)
        loss_min = np.min(batch_losses_array)
        loss_max = np.max(batch_losses_array)

        grad_norm_mean = np.mean(batch_grad_norms_array)
        grad_norm_std = np.std(batch_grad_norms_array)

        return avg_loss, loss_std, loss_min, loss_max, grad_norm_mean, grad_norm_std
    
    def validate(self, epoch):
        """验证模型 - 计算完整的评估指标"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 用于存储所有预测和真实标签
        all_preds = []
        all_labels = []
        all_probs = []  # 用于ROC-AUC计算

        # 记录每个batch的验证损失
        batch_val_losses = []

        # 【新增】分布匹配指标的累积列表
        js_divergences = []
        cosine_similarities = []
        top3_distribution_matches = []
        brier_scores = []
        top3_accuracies = []
        adaptive_topk_distribution_matches = []
        adaptive_topk_accuracies = []
        avg_k_values = []
        all_dist_metrics = []

        # 记录验证集样本的详细信息
        val_sample_predictions = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [验证]')
            for idx, data in enumerate(pbar):
                # 前向传播
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

                ############################ 分类/回归 标签转换 ############################

                ############原有软标签############
                temperature = 0.02  # 温度参数（使得分布更平滑）
                class_label = F.softmax(-data.y / temperature, dim=0).unsqueeze(0)

                loss = F.kl_div(output, class_label, reduction='batchmean')
                ############原有软标签############
                total_loss += loss.item()

                # 新增：记录每个batch的损失
                batch_val_losses.append(loss.item())

                # 【新增】计算分布匹配指标 从log_softmax转换为概率
                pred_probs = torch.exp(output)  # [1, num_classes]
                true_probs = class_label  # [1, num_classes]

                dist_metrics = self.compute_distribution_metrics(pred_probs, true_probs)
                all_dist_metrics.append(dist_metrics)
                js_divergences.append(dist_metrics['js_divergence'])
                cosine_similarities.append(dist_metrics['cosine_similarity'])
                top3_distribution_matches.append(dist_metrics['top3_distribution_match'])
                brier_scores.append(dist_metrics['brier_score'])
                top3_accuracies.append(dist_metrics['top3_accuracy'])
                adaptive_topk_distribution_matches.append(dist_metrics['adaptive_topk_distribution_match'])
                adaptive_topk_accuracies.append(dist_metrics['adaptive_topk_accuracy'])
                avg_k_values.append(dist_metrics['avg_k_value'])

                # 计算准确率
                pred = output.argmax(dim=1)  # 预测的最佳方法索引, shape: [1]
                true_label = class_label.argmax(dim=1) 
                # true_label = data.y.argmin()
                correct += (pred == true_label).sum().item()
                total += 1  # 每次处理一个图

                ############################ 分类/回归 标签转换 ############################
                
                # 收集预测和标签用于后续指标计算
                all_preds.append(pred.cpu().numpy()[0])   # ！！！！！！！！！为什么要把pred搬回cpu
                all_labels.append(data.y.argmin().item())
                # 转换为概率
                probs = F.softmax(output, dim=1).cpu().numpy()[0]  # 从logits转换
                all_probs.append(probs)

                val_sample_predictions.append({
                    'dataset_index': int(self.val_indices[idx]),
                    'true_label': int(data.y.argmin().item()),
                    'predicted_label': int(pred.cpu().numpy()[0]),
                    'true_label_name': self.class_names[data.y.argmin().item()],
                    'predicted_label_name': self.class_names[pred.cpu().numpy()[0]],
                    'is_correct': bool(pred.cpu().numpy()[0] == data.y.argmin().item()),
                    'prediction_probs': probs.tolist(),
                    'num_nodes': int(data.x.shape[0]),
                    'num_edges': int(data.edge_index.shape[1]),
                })
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # 计算验证损失的统计信息
        batch_val_losses_array = np.array(batch_val_losses)
        val_loss_std = np.std(batch_val_losses_array)
        val_loss_min = np.min(batch_val_losses_array)
        val_loss_max = np.max(batch_val_losses_array)
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算其他评估指标
        # 1. 宏平均 F1-Score - 指定labels确保考虑所有类别
        macro_f1 = f1_score(all_labels, all_preds, labels=list(range(self.args.num_classes)), average='macro', zero_division=0) * 100
        
        # 2. 加权平均 F1-Score - 指定labels确保考虑所有类别
        weighted_f1 = f1_score(all_labels, all_preds, labels=list(range(self.args.num_classes)), average='weighted', zero_division=0) * 100
        
        # 3. ROC-AUC (OvR - One-vs-Rest)
        try:
            from sklearn.preprocessing import label_binarize
            
            # 检查是否所有类别都有样本
            unique_labels = np.unique(all_labels)
            unique_counts = np.unique(all_labels, return_counts=True)
            
            # 调试信息：只在第一个epoch打印
            if epoch == 1:
                print(f"\n  [调试] 验证集标签分布: {dict(zip(unique_counts[0], unique_counts[1]))}")
                print(f"  [调试] all_probs shape: {all_probs.shape}, all_labels shape: {all_labels.shape}")
            
            # 关键修复：将标签二值化为3个类别（即使验证集中某些类别没有样本）
            # 这样可以确保与 all_probs 的列数匹配
            y_true_binarized = label_binarize(all_labels, classes=list(range(self.args.num_classes)))
            
            # 如果验证集中缺少某些类别，给出警告
            if len(unique_labels) < self.args.num_classes:
                missing_classes = set(list(range(self.args.num_classes))) - set(unique_labels)
                if epoch == 1:
                    print(f"  ⚠ 警告: 验证集中缺少类别 {missing_classes}，ROC-AUC可能不够准确")
            
            # 手动计算每个类别的AUC，然后取平均
            from sklearn.metrics import roc_auc_score as roc_score
            auc_scores = []
            for i in range(self.args.num_classes):  # num_classes个类别
                # 只有当该类别在验证集中存在时才计算AUC
                if i in unique_labels:
                    try:
                        auc_i = roc_score(y_true_binarized[:, i], all_probs[:, i])
                        auc_scores.append(auc_i)
                    except:
                        pass  # 如果单个类别计算失败，跳过
            
            # 计算平均AUC
            if len(auc_scores) > 0:
                roc_auc = np.mean(auc_scores) * 100
            else:
                roc_auc = 0.0
                if epoch == 1:
                    print(f"  ⚠ 警告: 无法计算任何类别的AUC")
                    
        except Exception as e:
            # 如果计算失败，打印详细错误信息
            print(f"\n  ⚠ ROC-AUC计算失败: {type(e).__name__}: {str(e)}")
            print(f"  [调试] all_labels: {all_labels[:10]}... (共{len(all_labels)}个)")
            print(f"  [调试] all_probs sample: {all_probs[:3]}")
            print(f"  [调试] unique labels: {np.unique(all_labels, return_counts=True)}")
            roc_auc = 0.0
        
        # 4. 混淆矩阵 - 指定labels参数确保始终生成3x3矩阵
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(self.args.num_classes)))
        
        # 返回所有指标
        metrics = {
            'loss': avg_loss,
            'loss_std': val_loss_std,
            'loss_min': val_loss_min,
            'loss_max': val_loss_max,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'all_probs': all_probs,
            # 【新增】分布匹配指标
            'js_divergence': np.mean(js_divergences),
            'cosine_similarity': np.mean(cosine_similarities),
            'top3_distribution_match': np.mean(top3_distribution_matches),
            'brier_score': np.mean(brier_scores),
            'top3_accuracy': np.mean(top3_accuracies),
            'adaptive_topk_distribution_match': np.mean(adaptive_topk_distribution_matches), 
            'adaptive_topk_accuracy': np.mean(adaptive_topk_accuracies),  # 新增
            'avg_k_value': np.mean(avg_k_values),  # 新增：平均k值
            'all_dist_metrics': all_dist_metrics,  # 【新增】
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history
        }
        
        # 保存最新模型
        latest_path = os.path.join(self.save_path, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_path, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ 最佳模型已保存 (验证损失: {val_loss:.4f})")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_path, 'train_history.json')
        # 需要排除不能序列化的项
        serializable_history = {
            key: value for key, value in self.train_history.items()
            if key != 'confusion_matrices'  # 混淆矩阵不保存到JSON
        }
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
    
    def plot_confusion_matrix(self, conf_matrix, epoch):
        """绘制混淆矩阵热力图"""
        # 修改：使用实例变量
        class_names = self.class_names
        
        # 根据类别数量动态调整图表大小
        fig_size = max(10, len(class_names) * 1.5)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        
        # 使用更小的字体以适应更多类别
        annot_fontsize = 10 if len(class_names) <= 3 else 8
    
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': annot_fontsize})
        
        # 根据epoch参数设置标题和文件名
        if epoch == 'final':
            plt.title(f'Confusion Matrix (Final - Best Model)', fontsize=16, fontweight='bold')
            cm_path = os.path.join(self.save_path, f'confusion_matrix_final.png')
        else:
            plt.title(f'Confusion Matrix (Epoch {epoch})', fontsize=16, fontweight='bold')
            cm_path = os.path.join(self.save_path, f'confusion_matrix_epoch{epoch}.png')
        
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
    
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"  混淆矩阵已保存到: {cm_path}")
    
    def plot_roc_curves(self, all_labels, all_probs, n_classes, epoch=None):
        """绘制ROC曲线（One-vs-Rest）"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # 修改：使用实例变量
        class_names = self.class_names
        n_classes = len(class_names)
        
        # 将标签二值化 - 使用实际的类别数量
        classes_list = list(range(n_classes))
        y_true_bin = label_binarize(all_labels, classes=classes_list)
        
        # 如果只有两个类别，label_binarize 返回 (n_samples,) 而不是 (n_samples, 1)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin.reshape(-1, 1), y_true_bin.reshape(-1, 1)])
        
        # 检查验证集中实际存在的类别
        unique_labels = np.unique(all_labels)
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 根据类别数量调整图表大小
        fig_size = max(10, 8)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        
        # 为不同数量的类别定义颜色
        if n_classes <= 3:
            colors = ['blue', 'red', 'green']
        else:
            # 使用色彩映射生成更多颜色
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            try:
                # 只有当该类别在验证集中存在时才绘制ROC曲线
                if i in unique_labels:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
                else:
                    # 如果该类别不存在，绘制虚线表示无数据
                    plt.plot([], [], color=colors[i], lw=2, linestyle='--', alpha=0.3,
                            label=f'{class_names[i]} (无数据)')
            except Exception as e:
                print(f"  警告: 绘制类别 {class_names[i]} 的ROC曲线时出错: {e}")
                continue
        
        # 绘制对角线（随机猜测的基准）
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        
        if epoch == 'final':
            plt.title('ROC Curves (Final - Best Model)', fontsize=16, fontweight='bold')
            roc_path = os.path.join(self.save_path, 'roc_curve_final.png')
        else:
            plt.title(f'ROC Curves (Epoch {epoch})', fontsize=16, fontweight='bold')
            roc_path = os.path.join(self.save_path, f'roc_curve_epoch{epoch}.png')
        
        # 根据类别数量调整图例位置和字体大小
        legend_fontsize = 10 if n_classes <= 3 else 8
        plt.legend(loc="lower right", fontsize=legend_fontsize)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ROC曲线已保存到: {roc_path}")
    
    def save_classification_report(self, all_labels, all_preds):
        """保存详细的分类报告"""
        # 修改：使用实例变量
        class_names = self.class_names
        n_classes = len(class_names)
        
        # 生成sklearn的分类报告
        # 使用实际的类别数量
        labels_list = list(range(n_classes))
        report = classification_report(all_labels, all_preds, 
                                       labels=labels_list,
                                       target_names=class_names, 
                                       digits=4,
                                       zero_division=0)
        
        # 保存到文本文件
        report_path = os.path.join(self.save_path, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("模型评估 - 详细分类报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"类别数量: {n_classes}\n")
            f.write(f"类别名称: {', '.join(class_names)}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        print(f"  分类报告已保存到: {report_path}")
    
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60 + "\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            # train_loss = self.train_epoch(epoch)
            train_loss, train_loss_std, train_loss_min, train_loss_max, grad_norm_mean, grad_norm_std = self.train_epoch(epoch)

            
            # 验证 - 现在返回完整的metrics字典
            metrics = self.validate(epoch)
            
            # 提取指标
            val_loss = metrics['loss']
            val_loss_std = metrics['loss_std']     
            val_loss_min = metrics['loss_min']     
            val_loss_max = metrics['loss_max']     
            val_accuracy = metrics['accuracy']
            val_macro_f1 = metrics['macro_f1']
            val_weighted_f1 = metrics['weighted_f1']
            val_roc_auc = metrics['roc_auc']
            conf_matrix = metrics['confusion_matrix']
            # 【新增】提取分布匹配指标
            val_js_divergence = metrics['js_divergence']
            val_cosine_similarity = metrics['cosine_similarity']
            val_top3_distribution_match = metrics['top3_distribution_match']
            val_brier_score = metrics['brier_score']
            val_top3_accuracy = metrics['top3_accuracy']
            val_adaptive_topk_distribution_match = metrics['adaptive_topk_distribution_match']
            val_adaptive_topk_accuracy = metrics['adaptive_topk_accuracy']
            val_avg_k_value = metrics['avg_k_value']
            
            # 学习率调整
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_loss_std'].append(train_loss_std)
            self.train_history['train_loss_min'].append(train_loss_min)
            self.train_history['train_loss_max'].append(train_loss_max)
            self.train_history['grad_norm'].append(grad_norm_mean)    
            self.train_history['grad_norm_std'].append(grad_norm_std) 
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_loss_std'].append(val_loss_std)   
            self.train_history['val_loss_min'].append(val_loss_min)   
            self.train_history['val_loss_max'].append(val_loss_max)   
            self.train_history['val_accuracy'].append(val_accuracy)
            self.train_history['val_macro_f1'].append(val_macro_f1)
            self.train_history['val_weighted_f1'].append(val_weighted_f1)
            self.train_history['val_roc_auc'].append(val_roc_auc)
            self.train_history['lr'].append(current_lr)
            # 【新增】记录分布匹配指标
            self.train_history['val_js_divergence'].append(val_js_divergence)
            self.train_history['val_cosine_similarity'].append(val_cosine_similarity)
            self.train_history['val_top3_distribution_match'].append(val_top3_distribution_match)
            self.train_history['val_brier_score'].append(val_brier_score)
            self.train_history['val_top3_accuracy'].append(val_top3_accuracy)
            self.train_history['val_adaptive_topk_distribution_match'].append(val_adaptive_topk_distribution_match)
            self.train_history['val_adaptive_topk_accuracy'].append(val_adaptive_topk_accuracy)
            self.train_history['val_avg_k_value'].append(val_avg_k_value)
            
            print(f"\nEpoch {epoch}/{self.args.epochs} 总结:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  梯度范数: {grad_norm_mean:.4f} ± {grad_norm_std:.4f}")  # 【新增】
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_accuracy:.2f}%")
            print(f"  宏平均 F1-Score: {val_macro_f1:.2f}%")
            print(f"  加权平均 F1-Score: {val_weighted_f1:.2f}%")
            print(f"  ROC-AUC (OvR): {val_roc_auc:.2f}%")
            # 【新增】打印分布匹配指标
            print(f"  ---")
            print(f"  ***分布匹配指标***")
            print(f"  JS散度: {val_js_divergence:.4f} (越小越好)")
            print(f"  余弦相似度: {val_cosine_similarity:.2f}% (越大越好)")
            print(f"  Top-3分布匹配率: {val_top3_distribution_match:.2f}%")
            print(f"  Brier Score: {val_brier_score:.4f} (越小越好)")
            print(f"  Top-3准确率: {val_top3_accuracy:.2f}%")
            print(f"  *自适应Top-k分布匹配率: {val_adaptive_topk_distribution_match:.2f}% (平均k={val_avg_k_value:.1f})")
            print(f"  *自适应Top-k准确率: {val_adaptive_topk_accuracy:.2f}%")
            print(f"  ---")
            print(f"  学习率: {current_lr:.6f}")
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/loss_std": train_loss_std,
                "train/loss_min": train_loss_min,
                "train/loss_max": train_loss_max,
                "val/loss": val_loss,
                "val/loss_std": val_loss_std,
                "val/loss_min": val_loss_min,
                "val/loss_max": val_loss_max, 
                "train/grad_norm_mean": grad_norm_mean, 
                "train/grad_norm_std": grad_norm_std,   
                "val/accuracy": val_accuracy,
                "val/macro_f1": val_macro_f1,
                "val/weighted_f1": val_weighted_f1,
                "val/roc_auc": val_roc_auc,
                "learning_rate": current_lr,
                # 【新增】分布匹配指标
                "distribution/js_divergence": val_js_divergence,
                "distribution/cosine_similarity": val_cosine_similarity,
                "distribution/top3_match": val_top3_distribution_match,
                "distribution/brier_score": val_brier_score,
                "distribution/top3_accuracy": val_top3_accuracy,
                "distribution/adaptive_topk_distribution_match": val_adaptive_topk_distribution_match,
                "distribution/adaptive_topk_accuracy": val_adaptive_topk_accuracy,
                "distribution/avg_k_value": val_avg_k_value,
            }, step=epoch)

            
            # # 额外记录混淆矩阵到 wandb（可选）
            # wandb.log({
            #     "confusion_matrix": wandb.plot.confusion_matrix(
            #         probs=None,
            #         y_true=metrics['all_labels'],
            #         preds=metrics['all_preds'],
            #         class_names=self.class_names
            #     )
            # }, step=epoch)
            
            # # 打印混淆矩阵
            # print(f"\n  混淆矩阵:")
            # # 修改：使用实例变量并动态调整格式
            # class_names = self.class_names
            # n_classes = min(len(class_names), conf_matrix.shape[0])
            
            # # 计算最大类别名称长度，用于对齐
            # max_name_len = max(len(name) for name in class_names[:n_classes])
            # padding = max(max_name_len, 12)
            
            # # 打印表头
            # header_names = [name[:9].ljust(9) for name in class_names[:n_classes]]
            # header = " " * (padding + 10) + "预测: " + "  ".join(header_names)
            # print(header)
            
            # # 打印每一行
            # for i in range(n_classes):
            #     row_label = f"真实: {class_names[i]}".ljust(padding + 10)
            #     row_values = "  ".join([f"{conf_matrix[i, j]:6d}" for j in range(n_classes)])
            #     print(row_label + row_values)
            
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # 保存训练历史
            self.save_history()
            
            print("-" * 60)
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最高验证准确率: {max(self.train_history['val_accuracy']):.2f}%")
        print(f"最高宏平均F1: {max(self.train_history['val_macro_f1']):.2f}%")
        print(f"最高加权F1: {max(self.train_history['val_weighted_f1']):.2f}%")
        print(f"最高ROC-AUC: {max(self.train_history['val_roc_auc']):.2f}%")
        print(f"---")
        print(f"*****分布匹配指标最佳值*****")
        print(f"最低JS散度: {min(self.train_history['val_js_divergence']):.4f}")
        print(f"最高余弦相似度: {max(self.train_history['val_cosine_similarity']):.2f}%")
        print(f"最高Top-3分布匹配率: {max(self.train_history['val_top3_distribution_match']):.2f}%")
        print(f"最低Brier Score: {min(self.train_history['val_brier_score']):.4f}")
        print(f"最高Top-3准确率: {max(self.train_history['val_top3_accuracy']):.2f}%")
        print(f"最高自适应Top-k分布匹配率: {max(self.train_history['val_adaptive_topk_distribution_match']):.2f}% (平均k={max(self.train_history['val_avg_k_value']):.1f})")
        print(f"最高自适应Top-k准确率: {max(self.train_history['val_adaptive_topk_accuracy']):.2f}%")
        print(f"---")
        
        print(f"模型保存路径: {self.save_path}")
        print("="*60 + "\n")
        
        # 基于最佳模型生成最终评估图表
        print("\n" + "="*60)
        print("正在基于最佳模型生成最终评估图表...")
        print("="*60)
        best_checkpoint_path = os.path.join(self.save_path, 'checkpoint_best.pt')
        if os.path.exists(best_checkpoint_path):
            # 加载最佳模型
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
            print(f"\n✓ 已加载最佳模型:")
            print(f"  - 来自 Epoch {best_epoch}")
            print(f"  - 验证损失: {best_val_loss:.4f}")
            
            # 在验证集上重新评估
            print(f"\n正在使用最佳模型评估验证集...")
            final_metrics = self.validate(best_epoch)
            
            print(f"\n最佳模型的最终评估指标:")
            print(f"  - 准确率: {final_metrics['accuracy']:.2f}%")
            print(f"  - 宏平均 F1: {final_metrics['macro_f1']:.2f}%")
            print(f"  - 加权平均 F1: {final_metrics['weighted_f1']:.2f}%")
            print(f"  - ROC-AUC: {final_metrics['roc_auc']:.2f}%")
            
            # 生成最终的混淆矩阵
            print(f"\n正在生成最终混淆矩阵...")
            final_cm_path = self.plot_confusion_matrix(final_metrics['confusion_matrix'], 'final')
            print(f"✓ 最终混淆矩阵已保存至: {final_cm_path}")
            
            # 生成最终的ROC曲线
            print(f"\n正在生成最终ROC曲线...")
            final_roc_path = self.plot_roc_curves(final_metrics['all_labels'], 
                                                   final_metrics['all_probs'], 
                                                   'final')
            print(f"✓ 最终ROC曲线已保存至: {final_roc_path}")
            
            # 生成基于最佳模型的详细分类报告
            print("\n正在生成详细分类报告（基于最佳模型）...")
            self.save_classification_report(final_metrics['all_labels'], 
                                           final_metrics['all_preds'])
            print("✓ 分类报告已保存！")

            print("\n正在保存最优概率分布（用于下游任务）...")
            best_dist_results = self.save_best_probability_distribution(
                final_metrics['all_labels'],
                final_metrics['all_probs'],
                final_metrics['all_dist_metrics'],
                epoch='final'
            )
            if 'weighted_distribution' in best_dist_results:
                for i, (class_name, prob) in enumerate(zip(self.class_names, best_dist_results['weighted_distribution'])):
                    wandb.summary[f"best_distribution/{class_name}"] = prob

            val_predictions_detailed = []
            self.model.eval()
            with torch.no_grad():
                for idx, data in enumerate(self.val_loader):
                    output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                    pred = output.argmax(dim=1)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    
                    val_predictions_detailed.append({
                        'dataset_index': int(self.val_indices[idx]),
                        'true_label': int(data.y.argmin().item()),
                        'predicted_label': int(pred.cpu().numpy()[0]),
                        'true_label_name': self.class_names[data.y.argmin().item()],
                        'predicted_label_name': self.class_names[pred.cpu().numpy()[0]],
                        'is_correct': bool(pred.cpu().numpy()[0] == data.y.argmin().item()),
                        'prediction_probs': {
                            class_name: float(prob) 
                            for class_name, prob in zip(self.class_names, probs)
                        },
                        'num_nodes': int(data.x.shape[0]),
                        'num_edges': int(data.edge_index.shape[1]),
                    })
            
            # 保存到文件
            val_predictions_path = os.path.join(self.save_path, 'validation_predictions.json')
            with open(val_predictions_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'num_val_samples': len(val_predictions_detailed),
                    'predictions': val_predictions_detailed,
                }, f, indent=4, ensure_ascii=False)
            
            print(f"✓ 验证集预测结果已保存至: {val_predictions_path}")
            
            # 记录到wandb
            val_pred_table = wandb.Table(columns=[
                "Dataset_Index", "True_Label", "Predicted_Label", 
                "Is_Correct", "Num_Nodes", "Num_Edges"
            ])
            for pred in val_predictions_detailed:
                val_pred_table.add_data(
                    pred['dataset_index'],
                    pred['true_label_name'],
                    pred['predicted_label_name'],
                    pred['is_correct'],
                    pred['num_nodes'],
                    pred['num_edges']
                )
            wandb.log({"final/validation_predictions": val_pred_table})
            
            pred_artifact = wandb.Artifact('validation_predictions', type='predictions')
            pred_artifact.add_file(val_predictions_path)
            wandb.log_artifact(pred_artifact)
            
            print(f"✓ 验证集预测结果已记录到wandb")
            
            print("\n" + "="*60)
            print("finial evaluation completed! All charts generated.")
            print("="*60)
        else:
            print(f"⚠ warning: best model checkpoint file not found: {best_checkpoint_path}")
            print("using the results of the last epoch to generate the classification report...")
            self.save_classification_report(metrics['all_labels'], metrics['all_preds'])

        print("="*60 + "\n")

        # 记录最终的最佳指标
        wandb.summary["best_val_loss"] = self.best_val_loss
        wandb.summary["best_val_accuracy"] = max(self.train_history['val_accuracy'])
        wandb.summary["best_macro_f1"] = max(self.train_history['val_macro_f1'])
        wandb.summary["best_weighted_f1"] = max(self.train_history['val_weighted_f1'])
        wandb.summary["best_roc_auc"] = max(self.train_history['val_roc_auc'])
        wandb.summary["best_js_divergence"] = min(self.train_history['val_js_divergence'])
        wandb.summary["best_cosine_similarity"] = max(self.train_history['val_cosine_similarity'])
        wandb.summary["best_top3_distribution_match"] = max(self.train_history['val_top3_distribution_match'])
        wandb.summary["best_brier_score"] = min(self.train_history['val_brier_score'])
        wandb.summary["best_top3_accuracy"] = max(self.train_history['val_top3_accuracy'])
        wandb.summary["best_adaptive_topk_distribution_match"] = max(self.train_history['val_adaptive_topk_distribution_match'])
        wandb.summary["best_adaptive_topk_accuracy"] = max(self.train_history['val_adaptive_topk_accuracy'])
        wandb.summary["best_avg_k_value"] = max(self.train_history['val_avg_k_value'])

        print("\n正在上传图表到 wandb...")
        if os.path.exists(os.path.join(self.save_path, 'training_history.png')):
            wandb.log({"charts/training_history": wandb.Image(os.path.join(self.save_path, 'training_history.png'))})
        if os.path.exists(os.path.join(self.save_path, 'loss_curve.png')):
            wandb.log({"charts/loss_curve": wandb.Image(os.path.join(self.save_path, 'loss_curve.png'))})
        if os.path.exists(os.path.join(self.save_path, 'confusion_matrix_final.png')):
            wandb.log({"charts/confusion_matrix_final": wandb.Image(os.path.join(self.save_path, 'confusion_matrix_final.png'))})
        if os.path.exists(os.path.join(self.save_path, 'roc_curves_final.png')):
            wandb.log({"charts/roc_curves_final": wandb.Image(os.path.join(self.save_path, 'roc_curves_final.png'))})

        wandb.finish()
        print("✓ wandb 日志已保存")


def main():
    parser = argparse.ArgumentParser(description='图神经网络训练脚本 - FJSP问题')
    
    # 数据相关参数
    parser.add_argument('--fjs_root_path', type=str, required=True,
                        help='FJS文件根目录路径')
    parser.add_argument('--label_root_path', type=str, required=True,
                        help='标签文件根目录路径')
    parser.add_argument('--label_name', type=str, default='mean',
                        help='标签名称 (默认: mean)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集占比 (默认: 0.8)')
    parser.add_argument('--model_name', type=str, 
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f'模型名称，可选: {list(MODEL_REGISTRY.keys())}')
    
    # 模型相关参数
    parser.add_argument('--node_features', type=int, default=4,
                        help='节点特征维度 (默认: 4)')
    parser.add_argument('--edge_features', type=int, default=2,
                        help='边特征维度 (默认: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度 (默认: 64)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='与class_names数量对应：分类类别数！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 - 注意：由于图大小不一致，实际batch_size固定为1，此参数已废弃 (默认: 32)')
    parser.add_argument('--accumulation_steps', type=int, default=32,
                        help='梯度累积步数，模拟更大的batch size (默认: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='L2正则化系数 (默认: 5e-4)')
    parser.add_argument('--patience', type=int, default=6,
                        help='学习率衰减的耐心值')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔（批次） (默认: 10)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='模型保存间隔（轮次） (默认: 10)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录 (默认: ./checkpoints)')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

