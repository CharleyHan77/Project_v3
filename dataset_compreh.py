# dataset.py - 完整修改版本

from torch.utils.data import dataset
import os
import os.path
import json
import random
import numpy as np

from torch_geometric.data import Data as Graph
import torch
from torch.utils.data.dataset import random_split


class Dataset(dataset.Dataset):
    def __init__(self, fjs_root_path, label_root_path, *, 
                 label_name="mean", 
                 use_comprehensive_score=True,  # 新增：是否使用综合评分
                 score_config=None,  # 新增：评分配置
                 device=None, 
                 train_ratio=0.8, 
                 augment=False, 
                 aug_prob=0.5):
        self.fjs_root_path = fjs_root_path
        self.label_root_path = label_root_path
        self.label_name = label_name
        self.use_comprehensive_score = use_comprehensive_score
        self.device = device
        self.train_ratio = train_ratio
        self.augment = augment
        self.aug_prob = aug_prob
        self.mode = 'train'
        
        # 评分配置（可自定义权重）
        self.score_config = score_config or {
            # 性能指标权重（总和应该接近1.0）
            'weights': {
                'mean': 0.30,           # 平均性能
                'min': 0.15,            # 最佳性能
                'median': 0.10,         # 中位数性能
                'std': 0.10,            # 稳定性（标准差）
                'range': 0.05,          # 性能范围（max-min）
                'conv_avg': 0.20,       # 收敛速度（平均代数）
                'conv_std': 0.05,       # 收敛稳定性
                'early_improvement': 0.05,  # 初期改进能力
            },
            # 是否对所有方法进行归一化（True）还是使用原始值加权（False）
            'normalize': True,
            # 是否反转归一化（对于"越小越好"的指标）
            'inverse_for_minimization': True,
        }

        labels = self._get_files(self.label_root_path)
        self.data = []

        # 第一遍扫描：收集所有方法的统计信息用于归一化
        if self.use_comprehensive_score and self.score_config['normalize']:
            self.global_stats = self._collect_global_stats(labels)
        else:
            self.global_stats = None

        # 第二遍扫描：生成数据
        for label_path in labels:
            with open(label_path, "r") as f:
                label = json.load(f)
        
            # 兼容新旧两种格式
            if "instance_info" in label:
                instance_name = label["instance_info"]["file_name"]
                fjs_path = os.path.join(self.fjs_root_path, instance_name)
            else:
                if label.get("sub_directory"):
                    fjs_path = os.path.join(self.fjs_root_path, label["dataset"], 
                                          label["sub_directory"], label["instance"])
                else:
                    fjs_path = os.path.join(self.fjs_root_path, label["dataset"], 
                                          label["instance"])
        
            label_info = label["initialization_methods"]
            
            g = self._convert_fjs(fjs_path)
            
            # 使用综合评分或简单标签
            if self.use_comprehensive_score:
                scores = self._calculate_comprehensive_scores(label_info)
            else:
                # 原始简单标签
                scores = [
                    label_info["FIFO_SPT"]["makespan"]["values"][self.label_name],
                    label_info["MOPNR_SPT"]["makespan"]["values"][self.label_name],
                    label_info["MOPNR_EET"]["makespan"]["values"][self.label_name],
                    label_info["MWKR_SPT"]["makespan"]["values"][self.label_name],
                    label_info["MWKR_EET"]["makespan"]["values"][self.label_name]
                ]
            
            scores = self.apply_temperature_scaling(scores, temperature=0.5)
            g.y = torch.log(torch.tensor(scores, dtype=torch.float) + 1)
            
            if self.device is not None:
                g.to(self.device)
            self.data.append(g)

    def apply_temperature_scaling(self, scores, temperature=0.5):
        """拉大标签差距"""
        scores = np.array(scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        scores_scaled = np.power(scores_norm, 1.0 / temperature)
        scores_final = scores_scaled * (scores.max() - scores.min()) + scores.min()
        return scores_final.tolist()

    def _collect_global_stats(self, label_paths):
        """
        第一遍扫描：收集所有实例的所有方法的统计信息
        用于全局归一化
        """
        methods = ["FIFO_SPT", "MOPNR_SPT", "MOPNR_EET", "MWKR_SPT", "MWKR_EET"]
        
        # 初始化统计容器
        stats = {
            'mean': [], 'min': [], 'median': [], 'std': [], 'range': [],
            'conv_avg': [], 'conv_std': [], 'early_improvement': []
        }
        
        for label_path in label_paths:
            with open(label_path, "r") as f:
                label = json.load(f)
            
            label_info = label["initialization_methods"]
            
            for method in methods:
                if method not in label_info:
                    continue
                    
                makespan_data = label_info[method]["makespan"]
                
                # 收集各项指标
                stats['mean'].append(makespan_data["values"]["mean"])
                stats['min'].append(makespan_data["values"]["min"])
                stats['median'].append(makespan_data["values"]["median"])
                stats['std'].append(makespan_data["values"]["std"])
                stats['range'].append(makespan_data["values"]["max"] - makespan_data["values"]["min"])
                stats['conv_avg'].append(makespan_data["convergence"]["avg_generation"])
                stats['conv_std'].append(makespan_data["convergence"]["std_generation"])
                
                # 计算早期改进率
                early_improvement = self._calculate_early_improvement(
                    makespan_data.get("convergence_curves", [])
                )
                stats['early_improvement'].append(early_improvement)
        
        # 计算每个指标的全局最小值和最大值
        global_stats = {}
        for key in stats:
            values = np.array(stats[key])
            global_stats[key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return global_stats

    def _calculate_early_improvement(self, convergence_curves):
        """
        从收敛曲线计算早期改进能力
        early_improvement = (初始值 - 第10代) / 初始值的平均改进率
        """
        if not convergence_curves or len(convergence_curves) == 0:
            return 0.0
        
        improvements = []
        early_gen = min(10, len(convergence_curves[0]))  # 前10代或曲线长度
        
        for curve in convergence_curves:
            if len(curve) < 2:
                continue
            initial = curve[0]
            early = curve[min(early_gen - 1, len(curve) - 1)]
            if initial > 0:
                improvement_rate = (initial - early) / initial
                improvements.append(improvement_rate)
        
        return np.mean(improvements) if improvements else 0.0

    def _calculate_comprehensive_scores(self, label_info):
        """
        为所有初始化方法计算综合评分
        返回5个方法的综合得分列表
        """
        methods = ["FIFO_SPT", "MOPNR_SPT", "MOPNR_EET", "MWKR_SPT", "MWKR_EET"]
        scores = []
        
        # 首先提取所有方法的原始指标
        raw_metrics = {}
        for method in methods:
            if method not in label_info:
                # 如果方法不存在，使用一个较大的惩罚值
                scores.append(99999.0)
                continue
            
            makespan_data = label_info[method]["makespan"]
            
            raw_metrics[method] = {
                'mean': makespan_data["values"]["mean"],
                'min': makespan_data["values"]["min"],
                'median': makespan_data["values"]["median"],
                'std': makespan_data["values"]["std"],
                'range': makespan_data["values"]["max"] - makespan_data["values"]["min"],
                'conv_avg': makespan_data["convergence"]["avg_generation"],
                'conv_std': makespan_data["convergence"]["std_generation"],
                'early_improvement': self._calculate_early_improvement(
                    makespan_data.get("convergence_curves", [])
                )
            }
        
        # 如果使用归一化
        if self.score_config['normalize'] and self.global_stats:
            # 使用全局统计进行归一化
            normalized_metrics = {}
            for method in raw_metrics:
                normalized_metrics[method] = {}
                for metric_name, value in raw_metrics[method].items():
                    norm_value = self._normalize_metric(
                        value, 
                        self.global_stats[metric_name],
                        inverse=(metric_name != 'early_improvement')  # early_improvement越大越好
                    )
                    normalized_metrics[method][metric_name] = norm_value
            
            # 计算加权得分
            for method in methods:
                if method not in normalized_metrics:
                    scores.append(99999.0)
                    continue
                
                score = 0.0
                for metric_name, weight in self.score_config['weights'].items():
                    score += normalized_metrics[method][metric_name] * weight
                
                scores.append(score)
        else:
            # 不使用归一化，局部归一化
            scores = self._calculate_local_normalized_scores(raw_metrics, methods)
        
        return scores

    def _normalize_metric(self, value, stat_dict, inverse=True):
        """
        归一化单个指标值到[0, 1]区间
        
        Args:
            value: 原始值
            stat_dict: 包含 min, max 的字典
            inverse: 如果True，对于"越小越好"的指标，归一化后反转
                    (使得较小的原始值得到较大的归一化值)
        """
        min_val = stat_dict['min']
        max_val = stat_dict['max']
        
        if max_val - min_val < 1e-8:
            return 0.5  # 如果所有值相同，返回中间值
        
        # 标准归一化到[0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        
        # 对于"越小越好"的指标，反转使得小值得到高分
        if inverse:
            normalized = 1.0 - normalized
        
        return normalized

    def _calculate_local_normalized_scores(self, raw_metrics, methods):
        """
        在当前实例的所有方法之间进行局部归一化
        （不依赖全局统计）
        """
        # 收集当前实例的各指标的min/max
        local_stats = {key: {'min': float('inf'), 'max': float('-inf')} 
                      for key in self.score_config['weights'].keys()}
        
        for method in raw_metrics:
            for metric_name, value in raw_metrics[method].items():
                local_stats[metric_name]['min'] = min(local_stats[metric_name]['min'], value)
                local_stats[metric_name]['max'] = max(local_stats[metric_name]['max'], value)
        
        # 归一化并计算得分
        scores = []
        for method in methods:
            if method not in raw_metrics:
                scores.append(99999.0)
                continue
            
            score = 0.0
            for metric_name, weight in self.score_config['weights'].items():
                value = raw_metrics[method][metric_name]
                norm_value = self._normalize_metric(
                    value,
                    local_stats[metric_name],
                    inverse=(metric_name != 'early_improvement')
                )
                score += norm_value * weight
            
            scores.append(score)
        
        return scores

    def set_mode(self, mode):
        """设置数据集模式"""
        if mode not in ['train', 'eval']:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        self.mode = mode

    def __getitem__(self, i):
        data = self.data[i]
        
        if self.mode == 'train' and self.augment and random.random() < self.aug_prob:
            data = self._augment_graph(data)
        
        return data

    def __len__(self):
        return len(self.data)

    # ... existing code (数据增强、图转换等方法保持不变) ...
    
    def _augment_graph(self, data):
        """图数据增强"""
        augmented = data.clone()
        
        edge_attr = augmented.edge_attr.clone()
        time_mask = edge_attr[:, 0] == 0
        
        if time_mask.any():
            noise_std = 0.002 + random.random() * 0.001
            time_values = edge_attr[time_mask, 1]
            noise = torch.randn_like(time_values) * noise_std * time_values
            edge_attr[time_mask, 1] = time_values + noise
            edge_attr[time_mask, 1] = torch.clamp(edge_attr[time_mask, 1], min=0.1)
    
        augmented.edge_attr = edge_attr
        return augmented

    def get_best_method(self, i):
        """获取第i个样本的最佳方法索引"""
        return self.data[i].y.argmin().item()

    def get_num_nodes(self, i):
        """获取第i个样本的节点数"""
        return self.data[i].x.shape[0]

    def _get_files(self, path):
        rsp = []
        for p in os.scandir(path):
            if os.path.isdir(p):
                rsp += self._get_files(p)
            else:
                rsp.append(p)
        return rsp
    
    def _convert_fjs(self, fjs_path):
        with open(fjs_path, "r") as f:
            fjs_lines = f.readlines()

        first_line_values = list(fjs_lines[0].split())
        if len(first_line_values) == 3:
            job_num, machine_num, _ = first_line_values
        else:
            job_num, machine_num = first_line_values

        job_num = int(job_num)
        machine_num = int(machine_num)

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

        return graph
