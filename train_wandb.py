import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import random_split
import argparse
import os
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score,
    classification_report
)
import seaborn as sns
import wandb

from model import get_model, MODEL_REGISTRY # 支持模型注册选择
from dataset import Dataset


"""
训练说明：
- 任务类型：分类任务（选择最佳的初始化方法）
- 标签格式：[heuristic性能, mixed性能, random性能]
- 标签转换：将性能值转为类别（argmin，因为性能值越小越好）
- 损失函数：NLLLoss（配合模型的log_softmax输出）
- 评估指标：分类准确率
"""


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
            project="GNN for fjsp",  # 项目名称，可以自定义
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

        if args.num_classes == 8:
            self.class_names = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 
                               'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
        elif args.num_classes == 3:
            self.class_names = ['heuristic', 'mixed', 'random']
        else:
            # 对于其他类别数，使用通用名称
            self.class_names = [f'class_{i}' for i in range(args.num_classes)]
        
        # 加载数据集
        print(f"正在加载数据集...")
        full_dataset = Dataset(args.fjs_root_path, args.label_root_path, device="cuda")
        
        # 划分训练集和验证集
        train_size = int(args.train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        # 统计训练集和验证集的类别分布
        train_methods = [data.y.argmin().item() for data in self.train_dataset]
        val_methods = [data.y.argmin().item() for data in self.val_dataset]
        
        train_nodes = [data.x.shape[0] for data in self.train_dataset]
        val_nodes = [data.x.shape[0] for data in self.val_dataset]
        
        print(f"\n数据集划分统计:")
        print(f"  总样本数: {len(full_dataset)}")
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        
        print(f"\n训练集类别分布:")
        for method_id, method_name in enumerate(['heuristic', 'mixed', 'random']):
            count = train_methods.count(method_id)
            print(f"  {method_name}: {count} ({100*count/len(train_methods):.1f}%)")
        
        print(f"\n验证集类别分布:")
        for method_id, method_name in enumerate(['heuristic', 'mixed', 'random']):
            count = val_methods.count(method_id)
            print(f"  {method_name}: {count} ({100*count/len(val_methods):.1f}%)")
        
        print(f"\n训练集节点数统计: min={min(train_nodes)}, max={max(train_nodes)}, mean={np.mean(train_nodes):.1f}")
        print(f"验证集节点数统计: min={min(val_nodes)}, max={max(val_nodes)}, mean={np.mean(val_nodes):.1f}")
        
        print(f"\n训练模式: 单图训练 + 梯度累积({args.accumulation_steps}步)")
        print(f"等效批次大小: {args.accumulation_steps}")
        
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
            'train_loss_std': [],  # 新增：每个epoch内batch损失的标准差
            'train_loss_min': [],  # 新增：每个epoch内batch损失的最小值
            'train_loss_max': [],  # 新增：每个epoch内batch损失的最大值
            'val_loss_std': [],    # 新增：验证集每个batch损失的标准差
            'val_loss_min': [],    # 新增：验证集每个batch损失的最小值
            'val_loss_max': []     # 新增：验证集每个batch损失的最大值
        }
        
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
    
    def save_config(self):
        """保存训练配置"""
        config = vars(self.args)
        config_path = os.path.join(self.save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"配置已保存至: {config_path}")
    
    def train_epoch(self, epoch):
        """训练一个epoch - 每次处理一个图"""
        self.model.train()
        total_loss = 0
        accumulation_steps = self.args.accumulation_steps

        # 新增：记录每个batch的损失
        batch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [训练]')
        for batch_idx, data in enumerate(pbar):
            # 前向传播
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

            ############################ 分类/回归 标签转换 ############################
            
            # 将性能值标签转换为分类标签（选择性能最小的方法）
            # output shape: [1, num_classes] -> [1, 8]（8种初始化方法的概率分布）
            # label shape: [8] -> [FIFO_SPT, FIFO_EET, MOPNR_SPT, MOPNR_EET, LWKR_SPT, LWKR_EET, MWKR_SPT, MWKR_EET的性能]
            # class_label = data.y.argmin().unsqueeze(0)  # shape: [1]
            class_label = F.softmax(-data.y, dim=0).unsqueeze(0)
            
            # 计算分类损失 - 使用NLLLoss（配合模型的log_softmax输出）
            loss = F.kl_div(output, class_label, reduction='batchmean')

            ############################ 分类/回归 标签转换 ############################

            # 记录原始损失值（在梯度累积之前）
            batch_losses.append(loss.item())
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            # 每accumulation_steps步更新一次权重
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            
            # 更新进度条
            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        # 处理最后剩余的梯度
        if len(self.train_loader) % accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)

        # 新增：计算batch损失的统计信息
        batch_losses_array = np.array(batch_losses)
        loss_std = np.std(batch_losses_array)
        loss_min = np.min(batch_losses_array)
        loss_max = np.max(batch_losses_array)

        return avg_loss, loss_std, loss_min, loss_max
    
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

        # 新增：记录每个batch的验证损失
        batch_val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [验证]')
            for data in pbar:

                # 前向传播
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

                ############################ 分类/回归 标签转换 ############################
                
                # 将性能值标签转换为分类标签（选择性能最小的方法）
                # label shape: [3] -> [heuristic性能, mixed性能, random性能]
                # class_label = data.y.argmin().unsqueeze(0)  # shape: [1]
                class_label = F.softmax(-data.y, dim=0).unsqueeze(0) 

                # 计算分类损失
                # loss = F.nll_loss(output, class_label)
                loss = F.kl_div(output, class_label, reduction='batchmean')
                total_loss += loss.item()

                # 新增：记录每个batch的损失
                batch_val_losses.append(loss.item())

                # 计算准确率
                pred = output.argmax(dim=1)  # 预测的最佳方法索引, shape: [1]
                true_label = class_label.argmax(dim=1)  # 真实的最佳方法索引, shape: [1]
                # true_label = data.y.argmin()
                correct += (pred == true_label).sum().item()
                total += 1  # 每次处理一个图
                
                # 收集预测和标签用于后续指标计算
                all_preds.append(pred.cpu().numpy()[0])   # ！！！！！！！！！为什么要把pred搬回cpu
                all_labels.append(true_label.cpu().numpy()[0])
                # 从log_softmax转换为概率
                probs = torch.exp(output).cpu().numpy()[0]
                all_probs.append(probs)

                ############################ 分类/回归 标签转换 ############################

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # 新增：计算验证损失的统计信息
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
        macro_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2], average='macro', zero_division=0) * 100
        
        # 2. 加权平均 F1-Score - 指定labels确保考虑所有类别
        weighted_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2], average='weighted', zero_division=0) * 100
        
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
            y_true_binarized = label_binarize(all_labels, classes=[0, 1, 2])
            
            # 如果验证集中缺少某些类别，给出警告
            if len(unique_labels) < 3:
                missing_classes = set([0, 1, 2]) - set(unique_labels)
                if epoch == 1:
                    print(f"  ⚠ 警告: 验证集中缺少类别 {missing_classes}，ROC-AUC可能不够准确")
            
            # 手动计算每个类别的AUC，然后取平均
            from sklearn.metrics import roc_auc_score as roc_score
            auc_scores = []
            for i in range(3):  # 3个类别
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
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        
        # 返回所有指标
        metrics = {
            'loss': avg_loss,
            'loss_std': val_loss_std,      # 新增
            'loss_min': val_loss_min,      # 新增
            'loss_max': val_loss_max,      # 新增
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'all_probs': all_probs
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
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        # 配置matplotlib样式
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 创建3x2的子图布局（增加新指标）
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle('Training Process Monitor', fontsize=18, fontweight='bold')
        
        # 子图1: 训练损失和验证损失
        # axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        # axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        epoch_idx = np.arange(1, len(self.train_history['train_loss']) + 1)

        # 转换为numpy数组以便计算置信区间
        train_loss_array = np.array(self.train_history['train_loss'])
        val_loss_array = np.array(self.train_history['val_loss'])

        # 计算置信区间（±5%波动范围，可根据需要调整）
        train_loss_lower = train_loss_array * 0.65
        train_loss_upper = train_loss_array * 1.35
        val_loss_lower = val_loss_array * 0.65
        val_loss_upper = val_loss_array * 1.35
        # 绘制曲线和阴影
        axes[0, 0].plot(epoch_idx, train_loss_array, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].fill_between(epoch_idx, train_loss_lower, train_loss_upper, color='blue', alpha=0.15)
        
        axes[0, 0].plot(epoch_idx, val_loss_array, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].fill_between(epoch_idx, val_loss_lower, val_loss_upper, color='red', alpha=0.15)
        
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss vs Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(loc='upper right', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 验证准确率
        axes[0, 1].plot(epochs, self.train_history['val_accuracy'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 105])  # 设置y轴范围为0-105%
        
        # 添加最高准确率标注
        max_acc = max(self.train_history['val_accuracy'])
        max_acc_epoch = self.train_history['val_accuracy'].index(max_acc) + 1
        axes[0, 1].axhline(y=max_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {max_acc:.2f}%')
        axes[0, 1].legend(loc='lower right', fontsize=10)
        
        # 子图3: F1-Score（宏平均和加权平均）
        axes[1, 0].plot(epochs, self.train_history['val_macro_f1'], 'b-', 
                       label='Macro F1', linewidth=2, marker='s', markersize=4)
        axes[1, 0].plot(epochs, self.train_history['val_weighted_f1'], 'r-', 
                       label='Weighted F1', linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12)
        axes[1, 0].set_title('F1-Score (Macro & Weighted)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='lower right', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
        
        # 子图4: ROC-AUC
        axes[1, 1].plot(epochs, self.train_history['val_roc_auc'], 'purple', 
                       linewidth=2, marker='d', markersize=4)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('ROC-AUC (%)', fontsize=12)
        axes[1, 1].set_title('ROC-AUC (One-vs-Rest, Macro Avg)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])
        
        # 添加最高ROC-AUC标注
        if len(self.train_history['val_roc_auc']) > 0:
            max_auc = max(self.train_history['val_roc_auc'])
            max_auc_epoch = self.train_history['val_roc_auc'].index(max_auc) + 1
            axes[1, 1].axhline(y=max_auc, color='r', linestyle='--', alpha=0.5, label=f'Best: {max_auc:.2f}%')
            axes[1, 1].legend(loc='lower right', fontsize=10)
        
        # 子图5: 学习率变化
        axes[2, 0].plot(epochs, self.train_history['lr'], 'm-', linewidth=2)
        axes[2, 0].set_xlabel('Epoch', fontsize=12)
        axes[2, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[2, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2, 0].set_yscale('log')  # 使用对数刻度
        axes[2, 0].grid(True, alpha=0.3)
        
        # 子图6: 所有评估指标对比
        axes[2, 1].plot(epochs, self.train_history['val_accuracy'], 'g-', 
                       label='Accuracy', linewidth=2, marker='o', markersize=3)
        axes[2, 1].plot(epochs, self.train_history['val_macro_f1'], 'b-', 
                       label='Macro F1', linewidth=2, marker='s', markersize=3)
        axes[2, 1].plot(epochs, self.train_history['val_weighted_f1'], 'r-', 
                       label='Weighted F1', linewidth=2, marker='^', markersize=3)
        axes[2, 1].plot(epochs, self.train_history['val_roc_auc'], 'purple', 
                       label='ROC-AUC', linewidth=2, marker='d', markersize=3)
        axes[2, 1].set_xlabel('Epoch', fontsize=12)
        axes[2, 1].set_ylabel('Score (%)', fontsize=12)
        axes[2, 1].set_title('All Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
        axes[2, 1].legend(loc='lower right', fontsize=9)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim([0, 105])
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.save_path, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 训练历史图表已保存至: {plot_path}")
        plt.close()
        
        # 额外生成一个单独的损失曲线图（更大更清晰）
        # fig2, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, alpha=0.8)
        # ax.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss', linewidth=2.5, alpha=0.8)
        # ax.set_xlabel('Epoch', fontsize=14)
        # ax.set_ylabel('Loss', fontsize=14)
        # ax.set_title('Training and Validation Loss Curve', fontsize=16, fontweight='bold')
        # ax.legend(loc='upper right', fontsize=12)
        # ax.grid(True, alpha=0.3)
        
        # # 标注最低验证损失
        # min_val_loss = min(self.train_history['val_loss'])
        # min_val_loss_epoch = self.train_history['val_loss'].index(min_val_loss) + 1
        # ax.plot(min_val_loss_epoch, min_val_loss, 'r*', markersize=15, 
        #         label=f'Best Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})')
        # ax.legend(loc='upper right', fontsize=12)
        
        # loss_plot_path = os.path.join(self.save_path, 'loss_curve.png')
        # plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        # 增加阴影
        # 额外生成一个单独的损失曲线图（更大更清晰）
        fig2, ax = plt.subplots(figsize=(12, 6))              
        
        # 转换为numpy数组
        train_loss_array = np.array(self.train_history['train_loss'])
        train_loss_std = np.array(self.train_history['train_loss_std'])
        val_loss_array = np.array(self.train_history['val_loss'])
        val_loss_std = np.array(self.train_history['val_loss_std'])  # 新增
        
        # 使用实际的标准差作为阴影范围
        train_loss_lower = train_loss_array - train_loss_std
        train_loss_upper = train_loss_array + train_loss_std
        val_loss_lower = val_loss_array - val_loss_std      # 新增
        val_loss_upper = val_loss_array + val_loss_std      # 新增
        
        # 绘制训练损失曲线和阴影
        ax.plot(epoch_idx, train_loss_array, 'b-', label='Train Loss', linewidth=2.5, alpha=0.9)
        ax.fill_between(epoch_idx, train_loss_lower, train_loss_upper, 
                        color='blue', alpha=0.3, label='Train ±1σ')
        
        # 绘制验证损失曲线和阴影
        ax.plot(epoch_idx, val_loss_array, 'r-', label='Val Loss', linewidth=2.5, alpha=0.9)
        ax.fill_between(epoch_idx, val_loss_lower, val_loss_upper, 
                color='red', alpha=0.3, label='Val ±1σ')
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training and Validation Loss Curve with Batch Variance', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 标注最低验证损失
        min_val_loss = min(val_loss_array)
        min_val_loss_epoch = np.argmin(val_loss_array) + 1
        ax.plot(min_val_loss_epoch, min_val_loss, 'r*', markersize=15, 
                label=f'Best Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})')
        ax.legend(loc='upper right', fontsize=12)
        
        loss_plot_path = os.path.join(self.save_path, 'loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 损失曲线图已保存至: {loss_plot_path}")
        plt.close()
        
        # 生成一个单独的准确率曲线图
        fig3, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, self.train_history['val_accuracy'], 'g-', linewidth=2.5, marker='o', 
                markersize=5, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('Validation Accuracy Curve', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 标注最高准确率
        ax.plot(max_acc_epoch, max_acc, 'r*', markersize=15,
                label=f'Best Accuracy: {max_acc:.2f}% (Epoch {max_acc_epoch})')
        ax.legend(loc='lower right', fontsize=12)
        
        acc_plot_path = os.path.join(self.save_path, 'accuracy_curve.png')
        plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 准确率曲线图已保存至: {acc_plot_path}")
        plt.close()
        
        # 生成F1-Score对比图
        fig4, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, self.train_history['val_macro_f1'], 'b-', label='Macro F1', 
                linewidth=2.5, marker='s', markersize=5, alpha=0.8)
        ax.plot(epochs, self.train_history['val_weighted_f1'], 'r-', label='Weighted F1', 
                linewidth=2.5, marker='o', markersize=5, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('F1-Score (%)', fontsize=14)
        ax.set_title('F1-Score Comparison (Macro vs Weighted)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        f1_plot_path = os.path.join(self.save_path, 'f1_score_curve.png')
        plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ F1-Score曲线图已保存至: {f1_plot_path}")
        plt.close()
        
        # 生成ROC-AUC曲线图
        fig5, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, self.train_history['val_roc_auc'], 'purple', 
                linewidth=2.5, marker='d', markersize=5, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('ROC-AUC (%)', fontsize=14)
        ax.set_title('ROC-AUC Curve (One-vs-Rest)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        if len(self.train_history['val_roc_auc']) > 0:
            ax.plot(max_auc_epoch, max_auc, 'r*', markersize=15,
                    label=f'Best ROC-AUC: {max_auc:.2f}% (Epoch {max_auc_epoch})')
            ax.legend(loc='lower right', fontsize=12)
        
        auc_plot_path = os.path.join(self.save_path, 'roc_auc_curve.png')
        plt.savefig(auc_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC-AUC曲线图已保存至: {auc_plot_path}")
        plt.close()
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60 + "\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            # train_loss = self.train_epoch(epoch)
            train_loss, train_loss_std, train_loss_min, train_loss_max = self.train_epoch(epoch)

            
            # 验证 - 现在返回完整的metrics字典
            metrics = self.validate(epoch)
            
            # 提取指标
            val_loss = metrics['loss']
            val_loss_std = metrics['loss_std']      # 新增
            val_loss_min = metrics['loss_min']      # 新增
            val_loss_max = metrics['loss_max']      # 新增
            val_accuracy = metrics['accuracy']
            val_macro_f1 = metrics['macro_f1']
            val_weighted_f1 = metrics['weighted_f1']
            val_roc_auc = metrics['roc_auc']
            conf_matrix = metrics['confusion_matrix']
            
            # 学习率调整
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_loss_std'].append(train_loss_std)
            self.train_history['train_loss_min'].append(train_loss_min)
            self.train_history['train_loss_max'].append(train_loss_max)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_loss_std'].append(val_loss_std)      # 新增
            self.train_history['val_loss_min'].append(val_loss_min)      # 新增
            self.train_history['val_loss_max'].append(val_loss_max)      # 新增
            self.train_history['val_accuracy'].append(val_accuracy)
            self.train_history['val_macro_f1'].append(val_macro_f1)
            self.train_history['val_weighted_f1'].append(val_weighted_f1)
            self.train_history['val_roc_auc'].append(val_roc_auc)
            self.train_history['lr'].append(current_lr)
            
            # 打印统计信息
            print(f"\nEpoch {epoch}/{self.args.epochs} 总结:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_accuracy:.2f}%")
            print(f"  宏平均 F1-Score: {val_macro_f1:.2f}%")
            print(f"  加权平均 F1-Score: {val_weighted_f1:.2f}%")
            print(f"  ROC-AUC (OvR): {val_roc_auc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")

            # 记录到 wandb
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
                "val/accuracy": val_accuracy,
                "val/macro_f1": val_macro_f1,
                "val/weighted_f1": val_weighted_f1,
                "val/roc_auc": val_roc_auc,
                "learning_rate": current_lr,
            }, step=epoch)

            
            # 额外记录混淆矩阵到 wandb（可选）
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=metrics['all_labels'],
                    preds=metrics['all_preds'],
                    class_names=self.class_names
                )
            }, step=epoch)
            
            # 打印混淆矩阵
            print(f"\n  混淆矩阵:")
            # 修改：使用实例变量并动态调整格式
            class_names = self.class_names
            n_classes = min(len(class_names), conf_matrix.shape[0])
            
            # 计算最大类别名称长度，用于对齐
            max_name_len = max(len(name) for name in class_names[:n_classes])
            padding = max(max_name_len, 12)
            
            # 打印表头
            header_names = [name[:9].ljust(9) for name in class_names[:n_classes]]
            header = " " * (padding + 10) + "预测: " + "  ".join(header_names)
            print(header)
            
            # 打印每一行
            for i in range(n_classes):
                row_label = f"真实: {class_names[i]}".ljust(padding + 10)
                row_values = "  ".join([f"{conf_matrix[i, j]:6d}" for j in range(n_classes)])
                print(row_label + row_values)
            
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
        print(f"模型保存路径: {self.save_path}")
        print("="*60 + "\n")
        
        # 生成训练历史图表
        print("正在生成训练历史图表...")
        self.plot_training_history()
        print("图表生成完成！")
        
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
            
            print("\n" + "="*60)
            print("最终评估完成！所有图表已生成。")
            print("="*60)
        else:
            print(f"⚠ 警告: 未找到最佳模型检查点文件: {best_checkpoint_path}")
            print("将使用最后一个epoch的结果生成分类报告...")
            self.save_classification_report(metrics['all_labels'], metrics['all_preds'])

        print("="*60 + "\n")

        # 记录最终的最佳指标
        wandb.summary["best_val_loss"] = self.best_val_loss
        wandb.summary["best_val_accuracy"] = max(self.train_history['val_accuracy'])
        wandb.summary["best_macro_f1"] = max(self.train_history['val_macro_f1'])
        wandb.summary["best_weighted_f1"] = max(self.train_history['val_weighted_f1'])
        wandb.summary["best_roc_auc"] = max(self.train_history['val_roc_auc'])

        # 上传生成的图表到 wandb
        print("\n正在上传图表到 wandb...")
        if os.path.exists(os.path.join(self.save_path, 'training_history.png')):
            wandb.log({"charts/training_history": wandb.Image(os.path.join(self.save_path, 'training_history.png'))})
        if os.path.exists(os.path.join(self.save_path, 'loss_curve.png')):
            wandb.log({"charts/loss_curve": wandb.Image(os.path.join(self.save_path, 'loss_curve.png'))})
        if os.path.exists(os.path.join(self.save_path, 'confusion_matrix_final.png')):
            wandb.log({"charts/confusion_matrix_final": wandb.Image(os.path.join(self.save_path, 'confusion_matrix_final.png'))})
        if os.path.exists(os.path.join(self.save_path, 'roc_curves_final.png')):
            wandb.log({"charts/roc_curves_final": wandb.Image(os.path.join(self.save_path, 'roc_curves_final.png'))})

        # 关闭 wandb
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
    parser.add_argument('--num_classes', type=int, default=8,
                        help='分类类别数 (默认: 8)')
    
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
    parser.add_argument('--patience', type=int, default=10,
                        help='学习率衰减的耐心值 (默认: 10)')
    
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

