#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 > results/context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --algo_name fm --epochs $2 > results/context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --epochs $2 > results/context/outputs_$1/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM ORIGINAL"
python main.py --dataset $1 --algo_name deepfm --epochs $2 > results/context/outputs_$1/reindexed_$1_deepfm_epochs=$2.txt
echo "DONE DFM ORIGINAL"

