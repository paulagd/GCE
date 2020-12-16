#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs lr dropout


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --epochs $2 --lr $3 --dropout $4 > results/context/outputs_$1/BPR_reindexed_$1_mf_epochs=$2_lr=$3_do=$4.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --epochs $2 --lr $3 --dropout $4 > results/context/outputs_$1/BPR_reindexed_$1_fm_epochs=$2_lr=$3_do=$4.txt
echo "DONE FM REINDEXED"

echo "Starting GRAPH experiments..."
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --gce --epochs $2 --lr $3 --dropout $4 > results/context/outputs_$1/BPR_graph_$1_mf_epochs=$2_lr=$3_do=$4.txt
#echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --gce --epochs $2 --lr $3 --dropout $4 > results/context/outputs_$1/BPR_graph_$1_fm_epochs=$2_lr=$3_do=$4.txt
echo "DONE FM REINDEXED-GCE"

