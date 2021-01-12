#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs
echo "Starting GRAPH EXPERIMENTS WITH GENDER SIDE INFO ON X MATRIX..."
python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.005 --batch_size 256 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_mf_epochs=$2.txt
echo "DONE MF-GCE-GENDR SIDE-INFO"
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.15 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO"
#python main.py --dataset $1 --algo_name nfm --epochs $2 --lr 0.0001 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_nfm_epochs=$2.txt
#echo "DONE NFM-GCE-GENDR SIDE-INFO"
#python main.py --dataset $1 --algo_name deepfm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_deepfm_epochs=$2.txt
#echo "DONE DFM-GCE-GENDR SIDE-INFO"
