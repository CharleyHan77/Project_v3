#!/bin/bash

# 无标签测试集推理脚本

# 设置路径
CHECKPOINT_PATH="/workspace/Project_v3/checkpoints_nsga2/NNConv_Attention_Pooling/20260325_163750-0324nsga2/checkpoint_best.pt"
TEST_FJS_PATH="/workspace/Project_v3/test_experiments_nsga2/test_dataset/BehnkeGeiger+Kacme"
MODEL_NAME="NNConv_Attention_Pooling"

# 运行推理
python /workspace/Project_v3/test_experiments_nsga2/test_model.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --test_fjs_path ${TEST_FJS_PATH} \
    --model_name ${MODEL_NAME} \
    --node_features 4 \
    --edge_features 2 \
    --hidden_dim 64 \
    --num_classes 5 \
    --output_dir /workspace/Project_v3/test_experiments_nsga2/results
