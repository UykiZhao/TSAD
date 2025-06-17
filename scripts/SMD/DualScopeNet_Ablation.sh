#!/bin/bash

# DualScopeNet 消融实验脚本
# 在SMD数据集上运行所有消融实验

echo "Starting DualScopeNet Ablation Study on SMD Dataset"
echo "================================================="

# 创建结果目录
mkdir -p ./ablation_results

# 基础参数
BASE_PARAMS="--data SMD \
  --root_path ./data/SMD/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --d_model 256 \
  --d_ff 256 \
  --e_layers 3 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 3 \
  --anomaly_ratio 0.5"

# 1. 完整模型（基准）
echo "1. Running full model (baseline)..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_Full \
  --ablation_mode full \
  --des 'Full_Model' \
  $BASE_PARAMS

# 2. 不使用PKI（周期分支使用原始TimesNet）
echo "2. Running without PKI (using original TimesNet for periodic branch)..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_NoPKI \
  --ablation_mode no_pki \
  --des 'No_PKI' \
  $BASE_PARAMS

# 3. 不使用增强MSTC（非周期分支使用简单卷积）
echo "3. Running without enhanced MSTC..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_NoMSTC \
  --ablation_mode no_mstc \
  --des 'No_MSTC' \
  $BASE_PARAMS

# 4. 不使用自适应融合（使用简单融合）
echo "4. Running without adaptive fusion..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_NoAdaptiveFusion \
  --ablation_mode no_adaptive_fusion \
  --des 'No_Adaptive_Fusion' \
  $BASE_PARAMS

# 5. 不使用异常感知融合
echo "5. Running without anomaly-aware fusion..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_NoAnomalyFusion \
  --ablation_mode no_anomaly_fusion \
  --des 'No_Anomaly_Fusion' \
  $BASE_PARAMS

# 6. 只使用周期分支
echo "6. Running periodic branch only..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_PeriodicOnly \
  --ablation_mode periodic_only \
  --des 'Periodic_Only' \
  $BASE_PARAMS

# 7. 只使用非周期分支
echo "7. Running aperiodic branch only..."
python -u run_ablation.py \
  --is_training 1 \
  --model_id SMD_DualScopeNet_AperiodicOnly \
  --ablation_mode aperiodic_only \
  --des 'Aperiodic_Only' \
  $BASE_PARAMS

echo "================================================="
echo "Ablation study completed!"
echo "Results saved in result_anomaly_detection.txt"