#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
#--lr 0.01 --optimizer adam --problem_type point --loss_type CL --prepro 10core --test_method tloo --factors 16

#0) TEST NFM ML-1M
#
#1) ML-1M
# more epochs all models
#
#2) more candidates (not 99 but 3900 ml-1m and 92,800 for lastfm)
#
#3) ADD VALIDATION AND SEE HOW METRICS GOES UP
#
#4) ADD more baselines
#
#5) add context
#
#6) multihop 2 and 3


##sh run_one_model.sh dataset_name epochs algo_name

#echo "Starting EMBEDDING no context ORIGINAL EXPERIMENTS on $1..."
#python main.py --dataset $1 --algo_name $3 --context --reindex --epochs $2 > results/no_context/outputs_$1/original_$1_$3_epochs=$2.txt
#echo "DONE $1-$3 ORIGINAL"
#
echo "Starting EMBEDDING no context EXPERIMENTS on $1..."
python main.py --problem_type pair --loss BPR --dataset $1 --algo_name $3 --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_$3_epochs=$2.txt
echo "DONE $1-$3 REINDEXED"

echo "Starting GRAPH no context experiments on $1..."
python main.py --problem_type pair --loss BPR --dataset $1 --algo_name $3 --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_$3_epochs=$2.txt
echo "DONE $1-$3 REINDEXED-GCE"

echo "Starting CONTEXT EXTENDED experiments  on $1 ..."

echo "Starting EMBEDDING context EXPERIMENTS  on $1..."
python main.py --problem_type pair --loss BPR --dataset $1 --algo_name $3 --epochs $2 > results/context/outputs_$1/reindexed_$1_$3_epochs=$2.txt
echo "DONE $1-$3 REINDEXED"

echo "Starting GRAPH context experiments  on $1..."
python main.py --problem_type pair --loss BPR --dataset $1 --algo_name $3 --gce --epochs $2 > results/context/outputs_$1/graph_$1_$3_epochs=$2.txt
echo "DONE $1-$3 REINDEXED-GCE"


#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
