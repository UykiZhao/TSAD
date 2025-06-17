#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id MSL_Transformer \
  --model Transformer \
  --data MSL \
  --root_path ./data/MSL/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --e_layers 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 