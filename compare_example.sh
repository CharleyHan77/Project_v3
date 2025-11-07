#!/bin/bash

# 模型对比示例脚本

# 对比所有模型（使用每个模型的最新训练记录）
python compare_models.py \
    --checkpoint_dir ./checkpoints \
    --save_dir ./model_comparison

# 或者只对比指定的模型
# python compare_models.py \
#     --checkpoint_dir ./checkpoints \
#     --models DirectedGraphClassifier GATClassifier \
#     --save_dir ./model_comparison

# 如果想绘制每个模型的所有训练记录，添加 --use_all_runs
# python compare_models.py \
#     --checkpoint_dir ./checkpoints \
#     --use_all_runs \
#     --save_dir ./model_comparison
