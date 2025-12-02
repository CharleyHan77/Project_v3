#!/bin/bash

# 图神经网络训练示例脚本
# 使用方法: bash train_example_wandb.sh

# 设置数据路径（请根据实际情况修改）
FJS_ROOT_PATH="/workspace/Project_v3/dataset_mksp_mean_balance"
LABEL_ROOT_PATH="/workspace/Project_v3/init_validity_result_mksp_mean_balance"

# 选择模型（可手动指定模型名称）
# MODEL_NAME="NNConv_Mean_Pooling"
# MODEL_NAME="NNConv_Multi_Scale_Pooling"
# MODEL_NAME="NNConv_Attention_Pooling"
# MODEL_NAME="NNConv_Deep_Attention_Pooling"
# MODEL_NAME="NNConv_Set2Set_Pooling"
# MODEL_NAME="NNConv_Max_Pooling"
# MODEL_NAME="GINE_Mean_Pooling"
MODEL_NAME="GINE_Deep_Attention_Pooling"
# MODEL_NAME="Transformer_Mean_Pooling"

# wandb离线模型
# export WANDB_MODE=offline
# wandb sync wandb/离线运行目录

# 训练参数
EPOCHS=100
ACCUMULATION_STEPS=16  # 梯度累积步数，模拟batch_size=xx的效果
LEARNING_RATE=0.001
HIDDEN_DIM=128

# 运行训练
# 注意：由于图大小不一致，每次只训练一个图，通过梯度累积模拟批处理
python train_wandb.py \
    --fjs_root_path ${FJS_ROOT_PATH} \
    --label_root_path ${LABEL_ROOT_PATH} \
    --model_name ${MODEL_NAME} \
    --epochs ${EPOCHS} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --lr ${LEARNING_RATE} \
    --hidden_dim ${HIDDEN_DIM} \
    --train_ratio 0.8 \
    --save_interval 10 \
    --log_interval 5 \
    --save_dir ./checkpoints


# 训练指定一个模型
# bash train_example.sh

# 一键对比所有模型
# bash compare_example.sh