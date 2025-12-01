import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GlobalAttention, Set2Set
from torch_geometric.nn import NNConv, GINEConv, TransformerConv

class NNConv_Mean_Pooling(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # 使用 NNConv (MPNN) - 支持边特征
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        # 默认 aggr='add'更优
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        
        # 分类
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征更新
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        ################### 图级别聚合（平均池化）
        x = global_mean_pool(x, batch)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        # return x


class NNConv_Max_Pooling(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # 使用 NNConv (MPNN) - 支持边特征
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        
        # 分类
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征更新
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        ################### 图级别聚合（最大池化）
        x = global_max_pool(x, batch)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class NNConv_Multi_Scale_Pooling(torch.nn.Module):
    """多尺度池化：拼接三种池化结果"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # 使用 NNConv (MPNN) - 支持边特征
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        
        # 多尺度池化后特征维度是 hidden_dim * 3
        # 拼接了 mean, max, sum 三种池化结果
        self.fc1 = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征更新
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        ############## 多尺度池化：拼接三种池化结果
        x_mean = global_mean_pool(x, batch)  # 平均池化
        x_max = global_max_pool(x, batch)    # 最大池化
        x_sum = global_add_pool(x, batch)    # 求和池化
         # 拼接三种池化结果
        x = torch.cat([x_mean, x_max, x_sum], dim=1)  # [batch, hidden_dim*3]
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class NNConv_Attention_Pooling(torch.nn.Module):
    """注意力池化"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # 使用 NNConv (MPNN) - 支持边特征
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)

        ################## 注意力门控网络：学习节点重要性
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn)
        
        # 分类
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征更新
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        ################### 注意力池化 学习每个节点的重要性权重
        x = self.attention_pool(x, batch)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        # return F.log_softmax(x, dim=1)
        return x

class NNConv_Deep_Attention_Pooling(torch.nn.Module):
    """更深层GCN+注意力池化"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # 第1层
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),  # 适中的dropout
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        
        # 第2-3层
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv3 = NNConv(hidden_dim, hidden_dim, nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # 注意力池化
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn)
        
        # 分类器（简化版）
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 第1层
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.15, training=self.training)
        
        # 第2层（带残差）
        identity = x
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.15, training=self.training)
        x = x + identity
        
        # 第3层（带残差）
        identity = x
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.15, training=self.training)
        x = x + identity
        
        # 注意力池化
        x = self.attention_pool(x, batch)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.35, training=self.training)
        x = self.fc2(x)
        
        return x


class NNConv_Set2Set_Pooling(torch.nn.Module):
    """基于LSTM的高级池化，适合需要多次读取图信息的任务"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes, processing_steps=3):
        super().__init__()
        # processing_steps：控制 Set2Set 模型内部 LSTM 在读出阶段执行的迭代步数。即在生成最终的图级表示之前，进行多少步的"注意力细化
        
        # 使用 NNConv (MPNN) - 支持边特征
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)

        ############ Set2Set池化：输出维度是 2 * hidden_dim
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        
        # 分类
        self.fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # 节点特征更新
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # Set2Set池化
        x = self.set2set(x, batch)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class GINE_Mean_Pooling(torch.nn.Module):
    """使用 GINEConv：表达能力最强"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        # GINE 层
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_features, train_eps=True)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_features, train_eps=True)
        
        # 分类器
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# class GATv2_Classifier(torch.nn.Module):
#     """使用 GATv2Conv：注意力机制   当前PYG版本过低""" 
#     def __init__(self, node_features, edge_features, hidden_dim, num_classes):
#         super().__init__()
        
#         self.num_heads = 4
        
#         # GAT 层（多头注意力）
#         self.conv1 = GATv2Conv(
#             node_features, 
#             hidden_dim // self.num_heads,
#             heads=self.num_heads,
#             edge_dim=edge_features,
#             concat=True,
#             dropout=0.3
#         )
        
#         self.conv2 = GATv2Conv(
#             hidden_dim,
#             hidden_dim // self.num_heads,
#             heads=self.num_heads,
#             edge_dim=edge_features,
#             concat=True,
#             dropout=0.3
#         )
        
#         # 分类器
#         self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
#     def forward(self, x, edge_index, edge_attr, batch):
#         x = F.elu(self.conv1(x, edge_index, edge_attr))
#         x = F.elu(self.conv2(x, edge_index, edge_attr))
        
#         x = global_mean_pool(x, batch)
        
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)


class Transformer_Mean_Pooling(torch.nn.Module):
    """TransformerConv：类似 Transformer 的注意力机制"""
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        
        self.num_heads = 4
        
        # TransformerConv 层
        self.conv1 = TransformerConv(
            node_features,
            hidden_dim // self.num_heads,
            heads=self.num_heads,
            edge_dim=edge_features,
            dropout=0.3,
            beta=True  # skip connection
        )
        
        self.conv2 = TransformerConv(
            hidden_dim,
            hidden_dim // self.num_heads,
            heads=self.num_heads,
            edge_dim=edge_features,
            dropout=0.3,
            beta=True
        )
        
        # 分类器
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# ==================== 模型注册表 ====================
MODEL_REGISTRY = {
    'NNConv_Mean_Pooling': NNConv_Mean_Pooling,
    'NNConv_Multi_Scale_Pooling': NNConv_Multi_Scale_Pooling,
    'NNConv_Attention_Pooling': NNConv_Attention_Pooling,
    "NNConv_Deep_Attention_Pooling": NNConv_Deep_Attention_Pooling,
    'NNConv_Set2Set_Pooling': NNConv_Set2Set_Pooling,
    "NNConv_Max_Pooling": NNConv_Max_Pooling,
    "GINE_Mean_Pooling": GINE_Mean_Pooling,
    "Transformer_Mean_Pooling": Transformer_Mean_Pooling
}

def get_model(model_name, **kwargs):
    """
    获取模型实例的工厂函数
    
    Args:
        model_name: 模型名称（在MODEL_REGISTRY中注册的名称）
        **kwargs: 模型初始化参数
    
    Returns:
        模型实例
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)
    