#!/bin/bash
#SBATCH -J biregnet
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname
#1 ct mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py  --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1


#1 mr ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --lr 0.001 --component 1 --components 1 --lambda_consis 0.1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#205 ct-mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py  --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1


#205 mr-ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --lr 0.001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#500 ct-mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#500 mr-ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 1 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 2 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 3 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --fold 4 --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2attreg.py --lr 0.001 --component 500 --components 500 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

