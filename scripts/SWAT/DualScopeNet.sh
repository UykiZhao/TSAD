#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id SWAT_DualScopeNet \
  --model DualScopeNet \
  --data SWAT \
  --root_path ./data/SWaT/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 3 \
  --patience 3 \
  --anomaly_ratio 0.5