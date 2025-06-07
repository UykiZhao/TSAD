#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id PSM_TimesNet \
  --model TimesNet \
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
  --e_layers 2 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 