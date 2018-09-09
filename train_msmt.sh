#!/usr/bin/env zsh

source activate pytorch2.7

# python train_class.py  -d msmt17 -a resnet50 --max-epoch 200 --save-dir save_folder
# python train_class_triHard.py  -d msmt17 -a resnet50 --lr 0.01 --max-epoch 1000 --save-dir save_folder --cuhk03-labeled 
python train_alignedreid.py -d msmt17 -a resnet50 --lr 0.00005 --max-epoch 2000 --save-dir save_folde --train-batch 128 --num-instances 4 --stepsize 200 --gamma 0.5 
echo 'done.'
