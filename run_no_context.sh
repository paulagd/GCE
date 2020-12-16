#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original no context experiments $1 ..."

#python main.py --dataset $1 --algo_name mf --reindex --context --epochs $2 > results/no_context/outputs_$1/original_$1_mf_epochs=$2.txt
#echo "DONE MF ORIGINAL"
#python main.py --dataset $1 --algo_name fm --reindex --context --epochs $2 > results/no_context/outputs_$1/original_$1_fm_epochs=$2.txt
#echo "DONE FM ORIGINAL"
#python main.py --dataset $1 --algo_name nfm --reindex --context --epochs $2 > results/no_context/outputs_$1/original_$1_nfm_epochs=$2.txt
#echo "DONE NFM ORIGINAL"

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM REINDEXED"
python main.py --dataset $1 --algo_name deepfm --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_deepfm_epochs=$2.txt
echo "DONE DFM REINDEXED"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --algo_name mf --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name nfm --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_nfm_epochs=$2.txt
echo "DONE NFM REINDEXED-GCE"
python main.py --dataset $1 --algo_name deepfm --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_deepfm_epochs=$2.txt
echo "DONE DFM REINDEXED-GCE"