#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id PSM_Autoformer \
  --model Autoformer \
  --data PSM \
  --root_path ./data/PSM/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 3 \
  --patience 3 \
  --anomaly_ratio 1.0 