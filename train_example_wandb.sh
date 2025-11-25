#!/bin/bash

# 图神经网络训练示例脚本
# 使用方法: bash train_example_wandb.sh

# 启用 wandb 离线模式
# export WANDB_MODE=offline

# 设置数据路径（请根据实际情况修改）
FJS_ROOT_PATH="/workspace/Project_v3/dataset_new"
LABEL_ROOT_PATH="/workspace/Project_v3/init_validity_result_new"

# 选择模型（可手动指定模型名称）
# MODEL_NAME="NNConv_Mean_Pooling"
# MODEL_NAME="NNConv_Multi_Scale_Pooling"
# MODEL_NAME="NNConv_Attention_Pooling"
# MODEL_NAME="NNConv_Set2Set_Pooling"
# MODEL_NAME="NNConv_Max_Pooling"
# MODEL_NAME="GINE_Mean_Pooling"
# MODEL_NAME="Transformer_Mean_Pooling"
MODEL_NAME="NNConv_Deep_Mean_Pooling"


# 训练参数
EPOCHS=50
ACCUMULATION_STEPS=16  # 梯度累积步数，模拟batch_size=16的效果
LEARNING_RATE=0.0001
HIDDEN_DIM=64

# Focal Loss参数
USE_FOCAL_LOSS=True  # 启用开关Focal Loss
FOCAL_GAMMA=5.0

# 新增：数据增强和重采样参数
USE_AUGMENTATION=False  # 启用数据增强
AUGMENTATION_PROB=0.6  # 60%的样本会被增强
USE_WEIGHTED_SAMPLING=False  # 启用加权采样
SAMPLING_MULTIPLIER=3  # n倍过采样，n越大，少数类被采样的概率越高

# 综合评分系统参数
USE_COMPREHENSIVE_SCORE=True  # 是否使用综合评分
SCORE_NORMALIZE=True  # 是否全局归一化

# 综合评分权重配置（总和应接近1.0）
WEIGHT_MEAN=0.30              # 平均性能
WEIGHT_MIN=0.15               # 最佳性能
WEIGHT_MEDIAN=0.10            # 中位数性能
WEIGHT_STD=0.10               # 稳定性
WEIGHT_RANGE=0.05             # 性能范围
WEIGHT_CONV_AVG=0.20          # 收敛速度
WEIGHT_CONV_STD=0.05          # 收敛稳定性
WEIGHT_EARLY_IMPROVEMENT=0.05 # 早期改进能力

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
    --train_ratio 0.7 \
    --save_interval 10 \
    --log_interval 5 \
    --save_dir ./checkpoints \
    --use_focal_loss ${USE_FOCAL_LOSS} \
    --focal_gamma ${FOCAL_GAMMA} \
    --use_augmentation ${USE_AUGMENTATION} \
    --augmentation_prob ${AUGMENTATION_PROB} \
    --use_weighted_sampling ${USE_WEIGHTED_SAMPLING} \
    --sampling_multiplier ${SAMPLING_MULTIPLIER} \
    --use_comprehensive_score ${USE_COMPREHENSIVE_SCORE} \
    --score_normalize ${SCORE_NORMALIZE} \
    --weight_mean ${WEIGHT_MEAN} \
    --weight_min ${WEIGHT_MIN} \
    --weight_median ${WEIGHT_MEDIAN} \
    --weight_std ${WEIGHT_STD} \
    --weight_range ${WEIGHT_RANGE} \
    --weight_conv_avg ${WEIGHT_CONV_AVG} \
    --weight_conv_std ${WEIGHT_CONV_STD} \
    --weight_early_improvement ${WEIGHT_EARLY_IMPROVEMENT}


# 训练指定一个模型
# bash train_example.sh

# 一键对比所有模型
# bash compare_example.sh