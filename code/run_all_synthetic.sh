#!/bin/bash

set -x

num_examples=1000
num_unlabeled=20000
dropout=0.1
suffix=a
num_layers=3
save_root=save_root
mode=direct,denoise,pretrain,pretrain+compose
python run_synthetic.py --num_examples ${num_examples} --num_unlabeled ${num_unlabeled} --dropout ${dropout} --suffix ${suffix} --num_layers ${num_layers} --save_root ${save_root} --mode ${mode}
