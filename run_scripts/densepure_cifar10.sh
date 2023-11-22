#!/usr/bin/env bash
cd ..

sigma=$1
steps=$2
reverse_seed=$3
n=$4
classifier=$5

python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_$n-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/cifar10-densepure-sample_num_$n-noise_$sigma-$steps-steps-$reverse_seed \
--sigma $sigma \
--N $n \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 20 9980) \
--use_id \
--certify_mode purify \
--advanced_classifier $classifier \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed
