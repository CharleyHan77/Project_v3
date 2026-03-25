from torch.utils.data import dataset
import os
import os.path
import json

from torch_geometric.data import Data as Graph
import torch
from torch.utils.data.dataset import random_split


# ==================== Pareto排名多目标标签计算模块 ====================
# 该模块实现基于Pareto支配关系和拥挤距离的多目标优化标签生成
# 返回综合得分向量（越小越好），由训练代码进行温度软化
# ======================================================================

import numpy as np

def pareto_dominates(obj1, obj2):
    """
    判断 obj1 是否 Pareto 支配 obj2
    
    Pareto支配定义：
    - obj1 在所有目标上都不差于 obj2（<=）
    - obj1 在至少一个目标上严格优于 obj2（<）
    
    Parameters:
    -----------
    obj1, obj2 : array-like, shape (n_objectives,)
        目标值向量，所有目标都是最小化（值越小越好）
    
    Returns:
    --------
    bool : obj1 是否支配 obj2
    """
    obj1 = np.array(obj1)
    obj2 = np.array(obj2)
    better_or_equal_in_all = np.all(obj1 <= obj2)
    better_in_at_least_one = np.any(obj1 < obj2)
    return better_or_equal_in_all and better_in_at_least_one


def compute_pareto_rank(objectives):
    """
    计算每个解的 Pareto 排名
    
    排名定义：
    - rank = 被其他解支配的次数
    - rank = 0: 该解不被任何其他解支配（Pareto前沿）
    - rank 越小越好
    
    Parameters:
    -----------
    objectives : array-like, shape (n_solutions, n_objectives)
        每行是一个解的目标值向量
    
    Returns:
    --------
    ranks : np.ndarray, shape (n_solutions,)
        每个解的 Pareto 排名
    """
    objectives = np.array(objectives)
    n_solutions = objectives.shape[0]
    ranks = np.zeros(n_solutions, dtype=int)
    
    for i in range(n_solutions):
        dominated_count = 0
        for j in range(n_solutions):
            if i != j and pareto_dominates(objectives[j], objectives[i]):
                dominated_count += 1
        ranks[i] = dominated_count
    
    return ranks


def compute_crowding_distance(objectives):
    """
    计算每个解的拥挤距离（Crowding Distance）
    
    算法流程（NSGA-II标准实现）：
    1. 对每个目标维度，按该目标值对解进行排序
    2. 边界解（最小值和最大值）的距离设为无穷大
    3. 中间解的距离 = 相邻两个解在该目标上的差值 / 目标值范围
    4. 每个解的总拥挤距离 = 各目标维度距离之和
    
    Parameters:
    -----------
    objectives : array-like, shape (n_solutions, n_objectives)
        每行是一个解的目标值向量
    
    Returns:
    --------
    distances : np.ndarray, shape (n_solutions,)
        每个解的拥挤距离，值越大表示该解越"独特"
    """
    objectives = np.array(objectives, dtype=float)
    n_solutions, n_objectives = objectives.shape
    distances = np.zeros(n_solutions)
    
    for m in range(n_objectives):
        sorted_indices = np.argsort(objectives[:, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        obj_min = objectives[sorted_indices[0], m]
        obj_max = objectives[sorted_indices[-1], m]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        for i in range(1, n_solutions - 1):
            idx = sorted_indices[i]
            distances[idx] += (objectives[sorted_indices[i + 1], m] - 
                              objectives[sorted_indices[i - 1], m]) / obj_range
    
    return distances


def compute_pareto_ranking_score(label_info, rule_names, label_name="mean", 
                                   rank_weight=1.0, crowding_weight=0.3,
                                   scale_factor=100.0):
    """
    基于 Pareto 排名和拥挤距离计算多目标综合得分
    
    返回的是原始得分（越小越好），而不是概率分布
    训练代码会自动对得分进行温度软化： F.softmax(-score / temperature)
    
    主要步骤：
    1. 从 label_info 中提取每个规则的三个目标值
    2. 计算 Pareto 排名（rank越小越好）
    3. 计算拥挤距离（distance越大越好，代表多样性）
    4. 综合得分 = rank_weight * rank - crowding_weight * normalized_distance
       注意：得分越小越好（与makespan一致）
    5. 返回 log(score + 1)，与单目标格式保持一致
    
    Parameters:
    -----------
    label_info : dict
        标签信息字典，格式: {rule_name: {objective: {"values": {"mean": ...}}}}
    rule_names : list of str
        规则名称列表
    label_name : str, default="mean"
        使用的统计量名称 ("mean", "median", "min", etc.)
    rank_weight : float, default=1.0
        Pareto排名的权重（rank越小越好）
    crowding_weight : float, default=0.3
        拥挤距离的权重（distance越大越好，代表多样性）
    scale_factor : float, default=100.0
        缩放因子，确保得分值在合理范围内（类似makespan的量级）
    
    Returns:
    --------
    torch.Tensor, shape (len(rule_names),)
        综合得分向量（越小越好），格式为 log(score + 1)
        与单目标的 log(makespan + 1) 格式完全一致
    
    Notes:
    ------
    - 训练代码会对返回值进行温度软化： F.softmax(-y / temperature)
    - 负号确保得分小的规则概率高
    - 这与单目标流程完全一致
    """
    # 1. 提取每个规则的三个目标值
    objectives = []
    for rule_name in rule_names:
        try:
            makespan = label_info[rule_name]["makespan"]["values"][label_name]
            max_load = label_info[rule_name]["max_machine_load"]["values"][label_name]
            total_load = label_info[rule_name]["total_machine_load"]["values"][label_name]
            objectives.append([makespan, max_load, total_load])
        except KeyError as e:
            raise KeyError(f"规则 {rule_name} 缺少必要的目标值: {e}")
    
    objectives = np.array(objectives, dtype=float)
    n_rules = len(rule_names)
    
    # 2. 计算 Pareto 排名（rank越小越好）
    pareto_ranks = compute_pareto_rank(objectives)
    
    # 3. 计算拥挤距离（distance越大越好）
    crowding_distances = compute_crowding_distance(objectives)
    
    # 4. 处理拥挤距离中的无穷大值
    finite_distances = crowding_distances[np.isfinite(crowding_distances)]
    if len(finite_distances) > 0:
        max_finite_distance = np.max(finite_distances)
        crowding_distances[np.isinf(crowding_distances)] = max_finite_distance * 2.0
    else:
        crowding_distances[np.isinf(crowding_distances)] = 1.0
    
    # 5. 归一化拥挤距离到 [0, 1]
    distance_min = np.min(crowding_distances)
    distance_max = np.max(crowding_distances)
    if distance_max > distance_min:
        crowding_distances_norm = (crowding_distances - distance_min) / (distance_max - distance_min)
    else:
        crowding_distances_norm = np.ones(n_rules) * 0.5
    
    # 6. 计算综合得分（越小越好）
    # rank越小越好，distance越大越好（但要转换为"越小越好"）
    composite_scores = rank_weight * pareto_ranks - crowding_weight * crowding_distances_norm
    
    # 7. 缩放到合理范围（类似makespan的量级）
    # 先平移使所有值为正，再缩放
    min_score = np.min(composite_scores)
    composite_scores = composite_scores - min_score + 1.0  # 所有值>=1
    composite_scores = composite_scores * scale_factor
    
    # 8. 返回 log(score + 1)，与单目标格式一致
    # 训练代码会对此进行 F.softmax(-y / temperature)
    log_scores = torch.log(torch.tensor(composite_scores, dtype=torch.float32) + 1.0)
    
    return log_scores


# ==================== Pareto排名模块结束 ====================

class Dataset(dataset.Dataset):
    def __init__(self, fjs_root_path, label_root_path, *, label_name = "mean", device = None, train_ratio = 0.8):
        self.fjs_root_path = fjs_root_path
        self.label_root_path = label_root_path
        self.label_name = label_name
        self.device = device
        self.train_ratio = train_ratio

        labels = self._get_files(self.label_root_path)
        self.data = []
        self.filenames = []

        for label_path in labels:
            with open(label_path, "r") as f:
                label = json.load(f)
        
            # 兼容新旧两种格式
            if "instance_info" in label:
                # 新格式：dataset_new 和 init_validity_result_new
                # FJS文件直接在根目录，通过 instance_info.file_name 获取文件名
                instance_name = label["instance_info"]["file_name"]
                fjs_path = os.path.join(self.fjs_root_path, instance_name)
            else:
                # 旧格式：dataset 和 init_validity_result
                # FJS文件有子目录结构（如 Barnes/mt10c1.fjs）
                if label.get("sub_directory"):
                    fjs_path = os.path.join(self.fjs_root_path, label["dataset"], label["sub_directory"], label["instance"])
                else:
                    fjs_path = os.path.join(self.fjs_root_path, label["dataset"], label["instance"])
        
            label_info = label["initialization_methods"]

            # 添加文件存在性检查：只处理存在的FJS文件
            if not os.path.exists(fjs_path):
                print(f"* FJS文件不存在，跳过: {fjs_path}")
                continue
            
            g = self._convert_fjs(fjs_path)
            # ============== 多目标标签计算 ==============
            # 使用 Pareto 排名法计算基于三个目标的综合得分
            # 目标：makespan, max_machine_load, total_machine_load（均为最小化）
            # 
            # 重要：返回的是原始得分（越小越好），训练代码会自动进行温度软化
            # 训练逻辑：class_label = F.softmax(-data.y / temperature)
            
            rule_names = [
                "FIFO_SPT",
                "FIFO_EET",
                "MOPNR_SPT",
                "MOPNR_EET",
                "LWKR_SPT",
                #"LWKR_EET",
                "MWKR_SPT"
                #"MWKR_EET"
            ]
            
            try:
                g.y = compute_pareto_ranking_score(
                    label_info=label_info,
                    rule_names=rule_names,
                    label_name=self.label_name,
                    rank_weight=1.0,         # Pareto排名权重
                    crowding_weight=0.3,     # 拥挤距离权重（鼓励多样性）
                    scale_factor=100.0       # 缩放因子（使得分在合理范围）
                )
            except KeyError as e:
                print(f"⚠ 警告: 实例 {instance_name} 的标签数据不完整，跳过: {e}")
                continue
            except Exception as e:
                print(f"⚠ 错误: 实例 {instance_name} 标签计算失败: {e}")
                continue

            if self.device is not None:
                g.to(self.device)
            self.data.append(g)
            self.filenames.append(instance_name)
        # self._split_dataset()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    ############# 验证集分层采样 #############

    def get_best_method(self, i):
        """获取第i个样本的最佳方法（性能最小的方法索引）
        返回: 0=heuristic, 1=mixed, 2=random
        """
        return self.data[i].y.argmin().item()

    def get_num_nodes(self, i):
        """获取第i个样本的节点数"""
        return self.data[i].x.shape[0]

    # def get_stratification_groups(self):
    #     """获取分层标签（基于节点规模和最佳方法的组合）
    #     用于分层采样以确保训练集和验证集的分布一致
    #     """
    #     import numpy as np

    #     groups = []
    #     for i in range(len(self.data)):
    #         # 获取最佳方法
    #         best_method = self.get_best_method(i)

    #         # 获取节点数并分组（小、中、大）
    #         num_nodes = self.get_num_nodes(i)
    #         if num_nodes < 50:
    #             size_group = 0  # 小规模
    #         elif num_nodes < 100:
    #             size_group = 1  # 中等规模
    #         else:
    #             size_group = 2  # 大规模

    #         # 组合：size_group * 3 + best_method
    #         # 这样可以得到9个分层组（3种规模 × 3种最佳方法）
    #         groups.append(size_group * 3 + best_method)

    #     return groups

    ############# 验证集分层采样 #############


    # def _split_dataset(self):
    #     train_size = int(self.train_ratio * len(self))
    #     val_size = len(self) - train_size
    #     self.train, self.validate = random_split(self, [train_size, val_size])

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

        # 兼容两种格式（新的 .fjs 文件格式第一行只有 2个值）
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
        # edge_attr: [是否工艺节点之间的边，工艺节点到机器节点之间的边上的耗时]
        edge_attr = []

        for job_id in range(job_num):
            data_iter = iter(list(map(int, fjs_lines[job_id + 1].split())))
            operator_num = next(data_iter)
            for operator_id in range(operator_num):
                operator_node_id = len(X)
                X.append([0, 0, 0, 1]) # 工艺节点
                if operator_id == 0:
                    edge_index.append([0, operator_node_id]) # 虚拟头节点连接到首个工艺节点
                else:
                    edge_index.append([operator_node_id - 1, operator_node_id]) # 上个工艺节点连接到当前工艺节点
                edge_attr.append([1, 0]) # 工艺节点之间的边特征值
                operator_machine_num = next(data_iter)
                for operator_machine_id in range(operator_machine_num):
                    machine_id = next(data_iter) - 1
                    time = next(data_iter)
                    edge_index.append([machine_id + 2, operator_node_id])
                    edge_attr.append([0, time])

            edge_index.append([len(X) - 1, 1]) # 最后一个工艺节点连接到虚拟尾节点
            edge_attr.append([1, 0])

        graph = Graph(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float))

        return graph
        