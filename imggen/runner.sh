#!/bin/bash
set -x

#### main results

# Composed model
python run_composed.py --n_epochs 1000 --dropout_prob 0.1 --n_epochs_pi 2 --weight_decay 0.0001

# baseline (Direct) results
python run_direct.py --n_epochs 1000


#### Ablations

# vary the smoothing param
python run_composed.py --n_epochs 1000 --eval_only_pi --smoothing_param 0.1
python run_composed.py --n_epochs 1000 --eval_only_pi --smoothing_param 1.0
python run_composed.py --n_epochs 1000 --eval_only_pi --smoothing_param 4.0
python run_composed.py --n_epochs 1000 --eval_only_pi --smoothing_param 8.0


# vary L2 regularization
python run_composed.py --n_epochs 1000 --n_epochs_pi 2 --eval_only_pi --weight_decay 0.000001
python run_composed.py --n_epochs 1000 --n_epochs_pi 2 --eval_only_pi --weight_decay 0.00001
python run_composed.py --n_epochs 1000 --n_epochs_pi 2 --eval_only_pi --weight_decay 0.001
python run_composed.py --n_epochs 1000 --n_epochs_pi 2 --eval_only_pi --weight_decay 0.01
python run_composed.py --n_epochs 1000 --n_epochs_pi 2 --eval_only_pi --weight_decay 0.1

python run_direct.py --n_epochs 1000 --weight_decay 0.000001
python run_direct.py --n_epochs 1000 --weight_decay 0.00001
python run_direct.py --n_epochs 1000 --weight_decay 0.0001
python run_direct.py --n_epochs 1000 --weight_decay 0.001
python run_direct.py --n_epochs 1000 --weight_decay 0.01
python run_direct.py --n_epochs 1000 --weight_decay 0.1


# vary unlabeled data 

python run_composed.py --n_epochs 1000 --n_epochs_pi 100 --num_fonts_pi 5000
python run_composed.py --n_epochs 1000 --n_epochs_pi 1000 --num_fonts_pi 100

# vary labeled data 

echo "EXP: VARY LABELED DATA COMPOSED"
python run_composed.py --n_epochs 1000 --num_examples 500 --pi_restore_path models/composed_diff/pi_checkpoint.pt
python run_composed.py --n_epochs 1000 --num_examples 1500 --pi_restore_path models/composed_diff/pi_checkpoint.pt
python run_composed.py --n_epochs 1000 --num_examples 4500 --pi_restore_path models/composed_diff/pi_checkpoint.pt
echo "EXP: VARY LABELED DATA DIRECT"
python run_direct.py --n_epochs 1000 --num_examples 500
python run_direct.py --n_epochs 1000 --num_examples 1500
python run_direct.py --n_epochs 1000 --num_examples 4500

# change regularization type
echo "EXP: CHANGE REGULARIZATION"
python run_composed.py --n_epochs 1000 --dropout_prob 0.1 --pi_restore_path models/composed_diff/pi_checkpoint.pt --weight_decay 0.0
python run_composed.py --n_epochs 1000 --dropout_prob 0.2 --pi_restore_path models/composed_diff/pi_checkpoint.pt --weight_decay 0.0
python run_composed.py --n_epochs 1000 --dropout_prob 0.3 --pi_restore_path models/composed_diff/pi_checkpoint.pt --weight_decay 0.0
python run_composed.py --n_epochs 1000 --dropout_prob 0.4 --pi_restore_path models/composed_diff/pi_checkpoint.pt --weight_decay 0.0
python run_composed.py --n_epochs 1000 --dropout_prob 0.5 --pi_restore_path models/composed_diff/pi_checkpoint.pt --weight_decay 0.0

# change perturbation type
echo "EXP: CHANGE PERTURBATION"
python run_composed.py --n_epochs 1000 --perturbation_type contrast --contrast_factor 0.5
python run_composed.py --n_epochs 1000 --perturbation_type emboss

