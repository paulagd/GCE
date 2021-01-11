#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.0001 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0 --not_early_stopping > results/context/outputs_$1/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM REINDEXED"
python main.py --dataset $1 --algo_name deepfm --epochs $2 --lr 0.001 --batch_size 2048 --dropout 0 --not_early_stopping > results/context/outputs_$1/reindexed_$1_deepfm_epochs=$2.txt
echo "DONE DFM REINDEXED"

echo "Starting GRAPH EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.0001 --batch_size 256 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_mf_epochs=$2.txt
echo "DONE MF-GCE"
python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.15 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_fm_epochs=$2.txt
echo "DONE FM-GCE"
python main.py --dataset $1 --algo_name nfm --epochs $2 --lr 0.0001 --batch_size 2048 --dropout 0 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_nfm_epochs=$2.txt
echo "DONE NFM-GCE"
python main.py --dataset $1 --algo_name deepfm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_deepfm_epochs=$2.txt
echo "DONE DFM-GCE"