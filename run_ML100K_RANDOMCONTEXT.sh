#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_ML100K_RANDOMCONTEXT.sh ml-100k 100

echo "Starting RANDOM CONTEXT EXPERIMENTS..."

#echo "Starting UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --uii --random_context --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UII-reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UII"
#python main.py --dataset $1 --algo_name fm --uii --random_context --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/randomC_UII-reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UII"
#
echo "Starting UIC EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --random_context --epochs $2 --lr 0.001 --batch_size 512 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/randomC_UIC_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --random_context --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/randomC_UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"

echo "Starting RANDOM CONTEXT EXPERIMENTS with INIT WEIGHTS LOADED..."

echo "Starting UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --uii --load_init_weights --random_context --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UII-reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UII"
#python main.py --dataset $1 --algo_name fm --uii --load_init_weights --random_context --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UII-reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UII"

echo "Starting UIC EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --load_init_weights --random_context --epochs $2 --lr 0.001 --batch_size 512 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UIC_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --load_init_weights --random_context --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"


#echo "Starting GRAPH EXPERIMENTS..."
#
#echo "Starting GCE-UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --random_context --uii --gce --epochs $2 --lr 0.005 --batch_size 256 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UII-graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UII" #done
#python main.py --dataset $1 --algo_name fm --random_context --uii --gce --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UII-graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UII"  # doing script
#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --random_context --gce --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UIC"  # doing script
#python main.py --dataset $1 --algo_name fm --random_context --gce --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UIC"  # doing script

#echo "Starting GCE-UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --random_context --load_init_weights --uii --gce --epochs $2 --lr 0.005 --batch_size 256 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UII-graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UII" #done
#python main.py --dataset $1 --algo_name fm --random_context --load_init_weights --uii --gce --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UII-graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UII"  # doing script
#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --random_context --load_init_weights --gce --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UIC"  # doing script
#python main.py --dataset $1 --algo_name fm --random_context --load_init_weights --gce --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/randomC_UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UIC"  # doing script

