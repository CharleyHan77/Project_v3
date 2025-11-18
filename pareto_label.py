import numpy as np
from typing import List, Dict, Tuple

class ParetoLabelGenerator:
    """
    基于Pareto支配关系的多目标标签生成器
    """
    
    def __init__(self, objectives: List[str] = None):
        """
        初始化
        
        Args:
            objectives: 目标列表，默认为['makespan', 'mean_flow_time', 'max_machine_load', 'total_machine_load']
        """
        if objectives is None:
            self.objectives = ['makespan', 'mean_flow_time', 'max_machine_load', 'total_machine_load']
        else:
            self.objectives = objectives
    
    def is_dominated(self, solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
        """
        判断solution_a是否被solution_b支配（所有目标都是最小化）
        
        支配定义：b支配a <=> 对所有目标i, b[i] <= a[i] 且至少存在一个目标j使得 b[j] < a[j]
        
        Args:
            solution_a: 解a的目标值数组 [obj1, obj2, obj3, obj4]
            solution_b: 解b的目标值数组
            
        Returns:
            True 如果a被b支配
        """
        # b在所有目标上都不差于a
        all_not_worse = np.all(solution_b <= solution_a)
        # b在至少一个目标上严格优于a
        at_least_one_better = np.any(solution_b < solution_a)
        
        return all_not_worse and at_least_one_better
    
    def calculate_pareto_ranks(self, methods_objectives: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """
        方案1: 计算Pareto支配等级（非支配排序）
        
        返回每个方法的Pareto等级：
        - 等级0：Pareto前沿（非支配解）
        - 等级1：去掉等级0后的Pareto前沿
        - 以此类推
        
        Args:
            methods_objectives: {方法名: {目标名: 目标值}}
            
        Returns:
            {方法名: Pareto等级}
        """
        methods = list(methods_objectives.keys())
        n = len(methods)
        
        # 转换为numpy数组便于计算
        objectives_matrix = np.array([
            [methods_objectives[method][obj] for obj in self.objectives]
            for method in methods
        ])
        
        # 记录每个解的等级
        ranks = {}
        remaining_indices = list(range(n))
        current_rank = 0
        
        while remaining_indices:
            # 在剩余解中找到非支配解（当前Pareto前沿）
            current_front = []
            
            for i in remaining_indices:
                is_dominated_by_any = False
                for j in remaining_indices:
                    if i != j and self.is_dominated(objectives_matrix[i], objectives_matrix[j]):
                        is_dominated_by_any = True
                        break
                
                if not is_dominated_by_any:
                    current_front.append(i)
            
            # 分配等级
            for idx in current_front:
                ranks[methods[idx]] = current_rank
                remaining_indices.remove(idx)
            
            current_rank += 1
        
        return ranks
    
    def calculate_domination_count(self, methods_objectives: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """
        方案2: 计算被支配次数
        
        返回每个方法被其他方法支配的次数（越小越好）
        
        Args:
            methods_objectives: {方法名: {目标名: 目标值}}
            
        Returns:
            {方法名: 被支配次数}
        """
        methods = list(methods_objectives.keys())
        n = len(methods)
        
        objectives_matrix = np.array([
            [methods_objectives[method][obj] for obj in self.objectives]
            for method in methods
        ])
        
        domination_count = {}
        for i, method_i in enumerate(methods):
            count = 0
            for j in range(n):
                if i != j and self.is_dominated(objectives_matrix[i], objectives_matrix[j]):
                    count += 1
            domination_count[method_i] = count
        
        return domination_count
    
    def calculate_hypervolume_contribution(self, methods_objectives: Dict[str, Dict[str, float]], 
                                          reference_point: np.ndarray = None) -> Dict[str, float]:
        """
        方案3: 计算超体积贡献
        
        超体积（Hypervolume）是多目标优化中常用的性能指标，
        表示Pareto前沿与参考点之间的空间体积。
        
        注意：精确计算超体积贡献在高维情况下复杂度很高，这里使用简化版本
        
        Args:
            methods_objectives: {方法名: {目标名: 目标值}}
            reference_point: 参考点（通常选择比所有解都差的点），如果为None则自动计算
            
        Returns:
            {方法名: 超体积贡献值}
        """
        methods = list(methods_objectives.keys())
        
        objectives_matrix = np.array([
            [methods_objectives[method][obj] for obj in self.objectives]
            for method in methods
        ])
        
        # 自动计算参考点（每个目标的最大值 + 10%）
        if reference_point is None:
            max_values = np.max(objectives_matrix, axis=0)
            reference_point = max_values * 1.1
        
        # 归一化目标值到[0, 1]区间
        min_values = np.min(objectives_matrix, axis=0)
        max_values = np.max(objectives_matrix, axis=0)
        normalized_matrix = (objectives_matrix - min_values) / (max_values - min_values + 1e-10)
        
        # 简化版：计算每个解到参考点的"矩形体积"
        contributions = {}
        for i, method in enumerate(methods):
            # 计算该解到参考点的体积（各维度差值的乘积）
            volume = np.prod(1.0 - normalized_matrix[i])  # 因为是最小化，所以用1.0减去归一化值
            contributions[method] = volume
        
        return contributions
    
    def calculate_distance_to_ideal(self, methods_objectives: Dict[str, Dict[str, float]], 
                                   normalized: bool = True) -> Dict[str, float]:
        """
        方案4: 计算到理想点的距离
        
        理想点是所有目标的最优值组成的点（通常不存在这样的解）
        距离越小表示综合性能越好
        
        Args:
            methods_objectives: {方法名: {目标名: 目标值}}
            normalized: 是否归一化目标值
            
        Returns:
            {方法名: 到理想点的欧氏距离}
        """
        methods = list(methods_objectives.keys())
        
        objectives_matrix = np.array([
            [methods_objectives[method][obj] for obj in self.objectives]
            for method in methods
        ])
        
        # 计算理想点（每个目标的最小值）
        ideal_point = np.min(objectives_matrix, axis=0)
        
        if normalized:
            # 归一化
            max_values = np.max(objectives_matrix, axis=0)
            objectives_matrix = (objectives_matrix - ideal_point) / (max_values - ideal_point + 1e-10)
            ideal_point = np.zeros_like(ideal_point)
        
        # 计算欧氏距离
        distances = {}
        for i, method in enumerate(methods):
            distance = np.linalg.norm(objectives_matrix[i] - ideal_point)
            distances[method] = distance
        
        return distances
    
    def generate_labels(self, methods_objectives: Dict[str, Dict[str, float]], 
                       strategy: str = 'pareto_rank',
                       inverse_for_label: bool = True) -> Dict[str, float]:
        """
        生成标签的统一接口
        
        Args:
            methods_objectives: {方法名: {目标名: 目标值}}
            strategy: 策略选择
                - 'pareto_rank': Pareto支配等级
                - 'domination_count': 被支配次数
                - 'hypervolume': 超体积贡献
                - 'distance': 到理想点的距离
            inverse_for_label: 是否反转标签（使得更好的解有更大的标签值）
            
        Returns:
            {方法名: 标签值}
        """
        if strategy == 'pareto_rank':
            labels = self.calculate_pareto_ranks(methods_objectives)
            # Pareto等级：0最好，需要反转
            if inverse_for_label:
                max_rank = max(labels.values())
                labels = {k: max_rank - v for k, v in labels.items()}
        
        elif strategy == 'domination_count':
            labels = self.calculate_domination_count(methods_objectives)
            # 被支配次数：越小越好，需要反转
            if inverse_for_label:
                max_count = max(labels.values())
                labels = {k: max_count - v for k, v in labels.items()}
        
        elif strategy == 'hypervolume':
            labels = self.calculate_hypervolume_contribution(methods_objectives)
            # 超体积贡献：越大越好，不需要反转
            if not inverse_for_label:
                max_hv = max(labels.values())
                labels = {k: max_hv - v for k, v in labels.items()}
        
        elif strategy == 'distance':
            labels = self.calculate_distance_to_ideal(methods_objectives)
            # 距离：越小越好，需要反转
            if inverse_for_label:
                max_dist = max(labels.values())
                labels = {k: max_dist - v for k, v in labels.items()}
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return labels


# ============ 使用示例 ============

# def example_usage():
#     """使用示例"""
    
#     # 模拟8种方法在4个目标上的性能
#     methods_objectives = {
#         'FIFO_SPT': {
#             'makespan': 100.0,
#             'mean_flow_time': 50.0,
#             'max_machine_load': 80.0,
#             'total_machine_load': 300.0
#         },
#         'FIFO_EET': {
#             'makespan': 95.0,
#             'mean_flow_time': 55.0,
#             'max_machine_load': 75.0,
#             'total_machine_load': 310.0
#         },
#         'MOPNR_SPT': {
#             'makespan': 105.0,
#             'mean_flow_time': 48.0,
#             'max_machine_load': 85.0,
#             'total_machine_load': 295.0
#         },
#         'MOPNR_EET': {
#             'makespan': 98.0,
#             'mean_flow_time': 52.0,
#             'max_machine_load': 78.0,
#             'total_machine_load': 305.0
#         },
#         'LWKR_SPT': {
#             'makespan': 102.0,
#             'mean_flow_time': 49.0,
#             'max_machine_load': 82.0,
#             'total_machine_load': 298.0
#         },
#         'LWKR_EET': {
#             'makespan': 97.0,
#             'mean_flow_time': 53.0,
#             'max_machine_load': 77.0,
#             'total_machine_load': 307.0
#         },
#         'MWKR_SPT': {
#             'makespan': 103.0,
#             'mean_flow_time': 51.0,
#             'max_machine_load': 83.0,
#             'total_machine_load': 300.0
#         },
#         'MWKR_EET': {
#             'makespan': 99.0,
#             'mean_flow_time': 54.0,
#             'max_machine_load': 79.0,
#             'total_machine_load': 306.0
#         }
#     }
    
#     generator = ParetoLabelGenerator()
    
#     print("=" * 60)
#     print("方案1: Pareto支配等级")
#     print("=" * 60)
#     pareto_ranks = generator.generate_labels(methods_objectives, strategy='pareto_rank')
#     for method, rank in sorted(pareto_ranks.items(), key=lambda x: x[1], reverse=True):
#         print(f"{method:15s}: {rank}")
    
#     print("\n" + "=" * 60)
#     print("方案2: 被支配次数")
#     print("=" * 60)
#     dom_counts = generator.generate_labels(methods_objectives, strategy='domination_count')
#     for method, count in sorted(dom_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"{method:15s}: {count}")
    
#     print("\n" + "=" * 60)
#     print("方案3: 超体积贡献")
#     print("=" * 60)
#     hv_contribs = generator.generate_labels(methods_objectives, strategy='hypervolume')
#     for method, hv in sorted(hv_contribs.items(), key=lambda x: x[1], reverse=True):
#         print(f"{method:15s}: {hv:.6f}")
    
#     print("\n" + "=" * 60)
#     print("方案4: 到理想点的距离")
#     print("=" * 60)
#     distances = generator.generate_labels(methods_objectives, strategy='distance')
#     for method, dist in sorted(distances.items(), key=lambda x: x[1], reverse=True):
#         print(f"{method:15s}: {dist:.6f}")


# if __name__ == "__main__":
#     example_usage()

