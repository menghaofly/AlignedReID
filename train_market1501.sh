#!/usr/bin/env sh

source activate pytorch2.7

# python train_class_sphere.py -d market1501 -a resnet50  --max-epoch 1000 --train-batch 256 --stepsize 200 --evaluate --resume log/best_model.pth.tar
python train_alignedreid_sphere.py -d market1501 -a resnet50  --max-epoch 1000 --train-batch 128 --stepsize 60 --lr 0.00005 --gamma 0.5
echo 'done.'
