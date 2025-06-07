#!/bin/bash

python -u run.py \
  --is_training 1 \
  --model_id SMD_TimesNet \
  --model TimesNet \
  --data SMD \
  --root_path ./data/SMD/ \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --e_layers 3 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --batch_size 128 \
  --anomaly_ratio 0.5 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 