#!/bin/bash

opt=$1
gpu=0
basename=`basename $opt`
expid=$(echo $basename | awk  '{ string=substr($0,1,3); print string; }')

echo "Started task, exp ${expid} on GPU no. ${gpu}"
echo $basename

CUDA_VISIBLE_DEVICES=$gpu nohup python -u train_mavoc.py --opt $opt > logs/train_${expid}_gpu${gpu}.log 2>&1 &
