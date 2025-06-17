#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id MSL_DualScopeNet \
  --model DualScopeNet \
  --data MSL \
  --root_path ./data/MSL/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --e_layers 3 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 3 \
  --anomaly_ratio 0.5 