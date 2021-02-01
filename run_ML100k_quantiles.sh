#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

echo "Starting QUANTILES  CONTEXT experiments on USERS $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --rankall --algo_name mf --remove_top_users $2 --remove_on user --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_reindexed_$1_mf.txt
#echo "DONE MF REINDEXED"
python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_reindexed_$1_fm.txt
echo "DONE FM REINDEXED"
#python main.py --dataset $1 --rankall --algo_name deepfm --remove_top_users $2 --epochs 100 --lr 0.0001 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_reindexed_$1_deepfm.txt
#echo "DONE DFM REINDEXED"
#
echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset $1 --rankall --algo_name mf --remove_top_users $2 --remove_on user --epochs 100 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_graph_$1_mf.txt
#echo "DONE MF-GCE"
python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_graph_$1_fm.txt
echo "DONE FM-GCE"
#python main.py --dataset $1 --rankall --algo_name deepfm --epochs 100 --lr 0.0001 --batch_size 256 --dropout 0 --gce --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_graph_$1_deepfm.txt
#echo "DONE DFM-GCE"


#echo "Starting GRAPH EXPERIMENTS WITH GENDER as SIDE INFO ON X MATRIX..."
#python main.py --dataset $1 --rankall --algo_name mf --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --side_information > results/context/outputs_$1/QUANTILE=$2_SI_UIC_graph_$1_mf.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"   # doing script
#python main.py --dataset $1 --rankall --algo_name fm --epochs 100 --lr 0.01 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/QUANTILE=$2_SI_UIC_graph_$1_fm.txt
#echo "DONE FM-GCE-gender SIDE-INFO"  # doing script
#python main.py --dataset $1 --rankall --algo_name deepfm --epochs 100 --lr 0.01 --batch_size 512 --dropout 0.15 --gce --not_early_stopping --side_information > results/context/outputs_$1/QUANTILE=$2_SI_UIC_graph_$1_deepfm.txt
#echo "DONE deepFM-GCE-GENDR SIDE-INFO"   # doing script
