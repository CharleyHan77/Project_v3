#!/bin/bash

# 无标签测试集推理脚本

# 设置路径
CHECKPOINT_PATH="/workspace/Project_v3/checkpoints_multiobjective/NNConv_Attention_Pooling/20251223_185718/checkpoint_best.pt"
TEST_FJS_PATH="/workspace/Project_v3/test_experiments_multiobjectice/test_dataset/Brandimarte"
MODEL_NAME="NNConv_Attention_Pooling"

# 运行推理
python /workspace/Project_v3/test_experiments_multiobjectice/test_model.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --test_fjs_path ${TEST_FJS_PATH} \
    --model_name ${MODEL_NAME} \
    --node_features 4 \
    --edge_features 2 \
    --hidden_dim 64 \
    --num_classes 6 \
    --output_dir /workspace/Project_v3/test_experiments_multiobjectice/results
