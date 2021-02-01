#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

echo "Starting original EXTENDED CONTEXT experiments $1 ..."


echo "Starting CONTEXT EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --rankall --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 256 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_reindexed_$1_mf.txt
#echo "DONE MF REINDEXED-UIC" #done
python main.py --dataset $1 --algo_name fm --rankall --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 2048 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_reindexed_$1_fm.txt
echo "DONE FM REINDEXED-UIC"  #done
####

echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --rankall --gce --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3__graph_$1_mf.txt
#echo "DONE MF GCE-UIC"
python main.py --dataset $1 --algo_name fm --rankall --gce --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.01 --batch_size 2048 --dropout 0 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3__graph_$1_fm.txt
echo "DONE FM GCE-UIC"
