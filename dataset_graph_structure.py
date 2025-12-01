import torch
import json
from torch_geometric.data import Data as Graph

def print_graph_structure(graph, fjs_path=None):
    """
    打印FJS析取图的详细结构信息
    
    参数:
        graph: torch_geometric.data.Data 对象
        fjs_path: FJS文件路径（可选，用于显示原始问题信息）
    """
    print("=" * 80)
    print("FJS析取图的GNN表征结构")
    print("=" * 80)
    
    # 1. 节点信息
    print("\n【1. 节点特征 (x)】")
    print(f"节点总数: {graph.x.shape[0]}")
    print(f"特征维度: {graph.x.shape[1]}")
    print(f"特征向量: [虚拟头节点, 虚拟尾节点, 机器节点, 工艺节点]")
    print("\n节点类型分布:")
    
    # 统计各类节点
    node_types = {
        "虚拟头节点": [],
        "虚拟尾节点": [],
        "机器节点": [],
        "工艺节点": []
    }
    
    for i, node_feat in enumerate(graph.x):
        if node_feat[0] == 1:
            node_types["虚拟头节点"].append(i)
        elif node_feat[1] == 1:
            node_types["虚拟尾节点"].append(i)
        elif node_feat[2] == 1:
            node_types["机器节点"].append(i)
        elif node_feat[3] == 1:
            node_types["工艺节点"].append(i)
    
    for node_type, indices in node_types.items():
        print(f"  {node_type}: {len(indices)}个 - 索引: {indices if len(indices) <= 10 else f'{indices[:10]}...'}")
    
    # 2. 边信息
    print("\n【2. 边索引 (edge_index)】")
    print(f"边的总数: {graph.edge_index.shape[1]}")
    print(f"边索引形状: {graph.edge_index.shape} (2 x num_edges)")
    print(f"边表示: edge_index[0]是源节点，edge_index[1]是目标节点")
    
    # 统计边的类型
    edge_types = {
        "头节点→首工艺": [],
        "工艺顺序边": [],
        "机器→工艺": [],
        "尾工艺→尾节点": []
    }
    
    for i in range(graph.edge_index.shape[1]):
        src = graph.edge_index[0, i].item()
        dst = graph.edge_index[1, i].item()
        edge_type = graph.edge_attr[i]
        
        if src == 0:  # 虚拟头节点
            edge_types["头节点→首工艺"].append((src, dst))
        elif dst == 1:  # 虚拟尾节点
            edge_types["尾工艺→尾节点"].append((src, dst))
        elif edge_type[0] == 1:  # 工艺顺序边
            edge_types["工艺顺序边"].append((src, dst))
        else:  # 机器→工艺
            edge_types["机器→工艺"].append((src, dst, edge_type[1].item()))
    
    print("\n边类型分布:")
    for edge_type, edges in edge_types.items():
        if edge_type == "机器→工艺":
            print(f"  {edge_type}: {len(edges)}条")
            if len(edges) <= 5:
                for src, dst, time in edges:
                    print(f"    机器节点{src} -> 工艺节点{dst}, 加工时间={time}")
            else:
                for src, dst, time in edges[:3]:
                    print(f"    机器节点{src} -> 工艺节点{dst}, 加工时间={time}")
                print(f"    ... (共{len(edges)}条边)")
        else:
            print(f"  {edge_type}: {len(edges)}条")
            if len(edges) <= 5:
                for edge in edges:
                    print(f"    {edge[0]} -> {edge[1]}")
    
    # 3. 边属性
    print("\n【3. 边属性 (edge_attr)】")
    print(f"边属性形状: {graph.edge_attr.shape}")
    print(f"属性维度: [是否工艺边(1=是,0=否), 加工时间(机器→工艺时有效)]")
    
    # 统计边属性
    precedence_edges = (graph.edge_attr[:, 0] == 1).sum().item()
    machine_edges = (graph.edge_attr[:, 0] == 0).sum().item()
    
    print(f"\n边属性统计:")
    print(f"  工艺顺序边 [1, 0]: {precedence_edges}条")
    print(f"  机器加工边 [0, time]: {machine_edges}条")
    
    if machine_edges > 0:
        machine_edge_times = graph.edge_attr[graph.edge_attr[:, 0] == 0, 1]
        print(f"    加工时间范围: [{machine_edge_times.min():.1f}, {machine_edge_times.max():.1f}]")
        print(f"    平均加工时间: {machine_edge_times.mean():.2f}")
    
    # 4. 标签信息（如果有）
    if hasattr(graph, 'y') and graph.y is not None:
        print("\n【4. 标签 (y) - 初始化方法的makespan预测】")
        print(f"标签形状: {graph.y.shape}")
        print(f"标签值 (log(makespan+1)): {graph.y}")
        print(f"对应makespan: {torch.exp(graph.y) - 1}")
    
    # 5. FJS原始信息
    if fjs_path:
        print(f"\n【5. 原始FJS文件信息】")
        print(f"文件路径: {fjs_path}")
    
    print("\n" + "=" * 80)


# 使用示例：
if __name__ == "__main__":
    # 加载dataset
    from dataset import Dataset
    
    fjs_root = "/workspace/Project_v3/dataset_new"
    label_root = "/workspace/Project_v3/init_validity_result_new"
    
    dataset = Dataset(
        fjs_root_path=fjs_root,
        label_root_path=label_root,
        label_name="mean"
    )
    
    print(f"数据集大小: {len(dataset)}\n")
    
    # 打印第一个图的结构
    if len(dataset) > 0:
        graph = dataset[0]
        print_graph_structure(graph)
        
        # 打印更详细的示例边
        print("\n【详细边示例（前20条）】")
        for i in range(min(20, graph.edge_index.shape[1])):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            attr = graph.edge_attr[i]
            print(f"边{i}: {src} -> {dst}, 属性: {attr.tolist()}")
            