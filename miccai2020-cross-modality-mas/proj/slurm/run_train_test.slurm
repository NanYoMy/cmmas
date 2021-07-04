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

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./training.py ./config/man_mr_ct_20_reg_one_ddf.ini ./config/man_mr_ct_20_votenet_one_ddf.ini
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./training.py ./config/man_ct_mr_20_reg_one_ddf.ini ./config/man_ct_mr_20_votenet_one_ddf.ini
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./training.py ./config/man_mr_ct_20_reg_mix.ini ./config/man_mr_ct_20_votenet_mix.ini
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./training.py ./config/man_ct_mr_20_reg_mix.ini ./config/man_ct_mr_20_votenet_mix.ini
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./inference_test_20.py ./config/man_mr_ct_20_reg_one_ddf.ini ./config/man_mr_ct_20_votenet_one_ddf.ini 24000
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./inference_test_20.py ./config/man_ct_mr_20_reg_one_ddf.ini ./config/man_ct_mr_20_votenet_one_ddf.ini 24000
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./inference_test_20.py ./config/man_mr_ct_20_reg_consistent.ini ./config/man_mr_ct_20_votenet_mix.ini 24000
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./inference_test_20.py ./config/man_ct_mr_20_reg_consistent.ini ./config/man_ct_mr_20_votenet_mix.ini 24000


