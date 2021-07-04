#!/bin/bash
#SBATCH -J label-reg-2
#SBATCH -p gpu
#SBATCH -w node02.chess
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 96:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./votenet_sample_generator_20.py ./config/man_ct_mr_20_reg.ini ./config/man_ct_mr_20_votenet.ini
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./sample_query_runner.py ./config/man_ct_mr_20_reg_bend.ini ./config/man_ct_mr_20_votenet.ini



