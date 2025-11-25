from torch.utils.data import dataset
import os
import os.path
import json
import random

from torch_geometric.data import Data as Graph
import torch
from torch.utils.data.dataset import random_split


class Dataset(dataset.Dataset):
    def __init__(self, fjs_root_path, label_root_path, *, label_name = "mean", device = None, train_ratio = 0.8, augment = False, aug_prob = 0.5):
        self.fjs_root_path = fjs_root_path
        self.label_root_path = label_root_path
        self.label_name = label_name
        self.device = device
        self.train_ratio = train_ratio
        self.augment = augment  # 是否启用数据增强（新增）
        self.aug_prob = aug_prob  # 数据增强概率（新增）
        self.mode = 'train'  # 数据集模式

        labels = self._get_files(self.label_root_path)
        self.data = []

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
            
            g = self._convert_fjs(fjs_path)
            g.y = torch.log(
                torch.tensor(
                    [
                        # label_info["FIFO_SPT"]["makespan"]["values"][self.label_name], 
                        # #label_info["FIFO_EET"]["makespan"]["values"][self.label_name], 
                        # label_info["MOPNR_SPT"]["makespan"]["values"][self.label_name], 
                        # label_info["MOPNR_EET"]["makespan"]["values"][self.label_name], 
                        # #label_info["LWKR_SPT"]["makespan"]["values"][self.label_name], 
                        # #label_info["LWKR_EET"]["makespan"]["values"][self.label_name], 
                        # label_info["MWKR_SPT"]["makespan"]["values"][self.label_name], 
                        # label_info["MWKR_EET"]["makespan"]["values"][self.label_name]
                        label_info["FIFO_SPT"]["max_machine_load"]["values"][self.label_name], 
                        label_info["FIFO_EET"]["max_machine_load"]["values"][self.label_name], 
                        label_info["MOPNR_SPT"]["max_machine_load"]["values"][self.label_name], 
                        label_info["MOPNR_EET"]["max_machine_load"]["values"][self.label_name], 
                        label_info["LWKR_SPT"]["max_machine_load"]["values"][self.label_name], 
                        label_info["LWKR_EET"]["max_machine_load"]["values"][self.label_name], 
                        label_info["MWKR_SPT"]["max_machine_load"]["values"][self.label_name], 
                        label_info["MWKR_EET"]["max_machine_load"]["values"][self.label_name]
                    ]
                ) + 1)
            if self.device is not None:
                g.to(self.device)
            self.data.append(g)

    def set_mode(self, mode):
        """设置数据集模式
        Args:
            mode: 'train' 或 'eval'
        """
        if mode not in ['train', 'eval']:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        self.mode = mode

    def __getitem__(self, i):
        data = self.data[i]
        
        # 只在训练模式下且启用增强时，随机决定要增强这个样本
        if self.mode == 'train' and self.augment and random.random() < self.aug_prob:
            data = self._augment_graph(data)
        
        return data

    def __len__(self):
        return len(self.data)


    # ============ 新增：数据增强方法 ============
    def _augment_graph(self, data):
        """
        图数据增强
        保持图的语义不变，只对特征进行合理扰动
        """
        # 深拷贝，避免修改原始数据
        augmented = data.clone()
        
        # 增强1：边时间特征添加高斯噪声（±5-10%）
        # 模拟加工时间的估计误差
        edge_attr = augmented.edge_attr.clone()
        time_mask = edge_attr[:, 0] == 0  # 只对机器-工序边（非工序连接边）添加噪声
        
        if time_mask.any():
            # 生成更小的噪声
            noise_std = 0.002 + random.random() * 0.001  # ← 2-3%，从5-10%降低
            time_values = edge_attr[time_mask, 1]
            noise = torch.randn_like(time_values) * noise_std * time_values
            edge_attr[time_mask, 1] = time_values + noise
            edge_attr[time_mask, 1] = torch.clamp(edge_attr[time_mask, 1], min=0.1)
    
        augmented.edge_attr = edge_attr
        
        # # 增强2：边Dropout（随机删除5-15%的机器-工序边）
        # # 模拟某些机器不可用的情况，增加模型鲁棒性
        # if random.random() < 0.4:  # 40%概率进行边dropout
        #     dropout_ratio = 0.05 + random.random() * 0.10  # 删除5-15%
        #     edge_keep_prob = 1.0 - dropout_ratio
        #     edge_mask = torch.rand(
        #         augmented.edge_index.shape[1], 
        #         device=augmented.edge_index.device
        #         ) < edge_keep_prob
            
        #     # 保留所有工序之间的连接边（这些边定义了工序顺序，不能删除）
        #     process_edges = augmented.edge_attr[:, 0] == 1
        #     edge_mask = edge_mask | process_edges
            
        #     # 应用mask
        #     augmented.edge_index = augmented.edge_index[:, edge_mask]
        #     augmented.edge_attr = augmented.edge_attr[edge_mask]
        
        # # 增强3：时间缩放（模拟不同速度的机器）
        # if random.random() < 0.2:  # 30%概率进行全局时间缩放
        #     scale_factor = 0.95 + random.random() * 0.15  # 0.9-1.1倍
        #     time_mask = augmented.edge_attr[:, 0] == 0
        #     augmented.edge_attr[time_mask, 1] *= scale_factor
        
        return augmented
    # ============ 数据增强方法结束 ============

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
        