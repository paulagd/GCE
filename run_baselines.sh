#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_baselines.sh dataset_name epochs

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

#echo "Starting original experiments $1 ..."

#python main.py --dataset $1 --algo_name mf --reindex  > outputs_ml-100k/original_$1_mf.txt
#echo "DONE MF ORIGINAL"
#python main.py --dataset $1 --algo_name fm --reindex > outputs_ml-100k/original_$1_fm.txt
#echo "DONE FM ORIGINAL"
#python main.py --dataset $1 --algo_name nfm --reindex > outputs_ml-100k/original_$1_nfm.txt
#echo "DONE NFM ORIGINAL"

echo "Starting EMBEDDING no context EXPERIMENTS on $1..."
python main.py --dataset $1 --algo_name mf --context --epochs $2 > results/no_context/outputs_ml-100k/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --context --epochs $2 > results/no_context/outputs_ml-100k/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --context --epochs $2 > results/no_context/outputs_ml-100k/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM ORIGINAL"

echo "Starting GRAPH no context experiments on $1..."
python main.py --dataset $1 --algo_name mf --gce --context --epochs $2 > results/no_context/outputs_ml-100k/graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce --context --epochs $2 > results/no_context/outputs_ml-100k/graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name nfm --gce --context --epochs $2 > results/no_context/outputs_ml-100k/graph_$1_nfm_epochs=$2.txt
echo "DONE NFM REINDEXED-GCE"



echo "Starting CONTEXT EXTENDED experiments  on $1 ..."

echo "Starting EMBEDDING context EXPERIMENTS  on $1..."
python main.py --dataset $1 --algo_name mf --epochs $2 > results/context/outputs_ml-100k/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --epochs $2 > results/context/outputs_ml-100k/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --epochs $2 > results/context/outputs_ml-100k/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM ORIGINAL"

echo "Starting GRAPH context experiments  on $1..."
python main.py --dataset $1 --algo_name mf --gce --epochs $2 > results/context/outputs_ml-100k/graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce --epochs $2 > results/context/outputs_ml-100k/graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name nfm --gce --epochs $2 > results/context/outputs_ml-100k/graph_$1_nfm_epochs=$2.txt
echo "DONE NFM REINDEXED-GCE"


#echo "Starting original experiments ..."
#
#python main.py --dataset $1 --algo_name mf --reindex > outputs/original/$1_mf.txt
#echo "DONE MF ORIGINAL"
#python main.py --dataset $1 --algo_name fm --reindex > outputs/original/$1_fm.txt
#echo "DONE FM ORIGINAL"
##python main.py --dataset $1 --algo_name neumf --reindex > outputs/original/$1_neufm.txt
##echo "DONE NEUMF ORIGINAL"
#python main.py --dataset $1 --algo_name nfm --reindex > outputs/original/$1_nfm.txt
#echo "DONE NFM ORIGINAL"
#
#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf > outputs/reindexed/$1_mf.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --algo_name fm > outputs/reindexed/$1_fm.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf > outputs/reindexed/neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm > outputs/reindexed/$1_nfm.txt
#echo "DONE NFM REINDEXED"

#echo "Starting GRAPH experiments..."
#python main.py --dataset $1 --algo_name mf --gce > outputs/graph/$1_mf.txt
#echo "DONE MF REINDEXED-GCE"
#python main.py --dataset $1 --algo_name fm --gce > outputs/graph/$1_fm.txt
#echo "DONE FM REINDEXED-GCE"
#python main.py --dataset ml-100k --algo_name neumf --gce > outputs/graph/$1_neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm  --gce > outputs/graph/$1_nfm.txt
#echo "DONE NFM REINDEXED-GCE"

#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
