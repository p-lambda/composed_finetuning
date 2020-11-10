#!/bin/bash

set -x

num_examples=1000
num_unlabeled=20000
dropout=0.1
suffix=a
python run_synthetic.py --num_examples ${num_examples} --num_unlabeled ${num_unlabeled} --dropout ${dropout} --suffix ${suffix}

# for num_examples in 500 1000 2000 do
#     for num_unlabeled in 5000 10000 20000 do
#         python run_synthetic.py --num_examples ${num_examples} --num_unlabeled ${num_unlabeled} --dropout ${dropout} --suffix ${suffix}
#     done
# done
# 
# num_examples=1000
# num_unlabeled=20000
# suffix=a
# for dropout in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 do
#         python run_synthetic.py --num_examples ${num_examples} --num_unlabeled ${num_unlabeled} --dropout ${dropout} --suffix ${suffix}
# done
