#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id SWAT_Informer \
  --model Informer \
  --data SWAT \
  --root_path ./data/SWAT/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 2 \
  --factor 5 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 