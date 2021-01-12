#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs



echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting NO CONTEXT EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --context --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --context --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"

echo "Starting UII EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --uii --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UII"
python main.py --dataset $1 --algo_name fm --uii --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UII"

echo "Starting UIC EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"


#echo "Starting GRAPH EXPERIMENTS..."
#
#echo "Starting GCE-UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --uii --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UII"
#python main.py --dataset $1 --algo_name fm --uii --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UII"
#
#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UIC"
#python main.py --dataset $1 --algo_name fm --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UIC"


#echo "Starting GRAPH EXPERIMENTS WITH GENDER SIDE INFO ON X MATRIX..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr ? --batch_size ? --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr ? --batch_size ? --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO"




#python main.py --dataset $1 --algo_name nfm --epochs $2 --lr 0.0001 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_nfm_epochs=$2.txt
#echo "DONE NFM-GCE-GENDR SIDE-INFO"
#python main.py --dataset $1 --algo_name deepfm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_graph_$1_deepfm_epochs=$2.txt
#echo "DONE DFM-GCE-GENDR SIDE-INFO"
