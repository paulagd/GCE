#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

echo "Starting original EXTENDED CONTEXT experiments $1 ..."

#echo "Starting NO CONTEXT EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --context --epochs $2 --lr 0.0005 --batch_size 256 --dropout 0.5 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED" #done
#python main.py --dataset $1 --algo_name fm --context --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED"  #done

echo "Starting CONTEXT EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --rankall --epochs $2 --lr 0.0005 --batch_size 256 --dropout 0.15 > results/context/outputs_$1/UIC_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC" #done
python main.py --dataset $1 --algo_name deepfm --rankall --epochs $2 --lr 0.01 --batch_size 512 --dropout 0.15 > results/context/outputs_$1/UIC_reindexed_$1_dfm_epochs=$2.txt
echo "DONE DEEPMF REINDEXED-UIC" #dones
python main.py --dataset $1 --algo_name fm --rankall --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0.15 > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"  #done
python main.py --dataset $1 --algo_name ncf --rankall --epochs $2 --lr 0.0001 --batch_size 512 --dropout 0.5 > results/context/outputs_$1/UIC_reindexed_$1_NCF_epochs=$2.txt
echo "DONE NCF REINDEXED-UIC"  #done


echo "Starting GRAPH EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --rankall --gce --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 > results/context/outputs_$1/UIC_graph_$1_mf_epochs=$2.txt
echo "DONE MF GCE-UIC"
python main.py --dataset $1 --algo_name deepfm --rankall --gce --epochs $2 --lr 0.01 --batch_size 256 --dropout 0.5 > results/context/outputs_$1/UIC_graph_$1_deepfm_epochs=$2.txt
echo "DONE deepfm GCE-UIC"  # done
python main.py --dataset $1 --algo_name fm --rankall --gce --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
echo "DONE FM GCE-UIC"
python main.py --dataset $1 --algo_name ncf --rankall --gce --epochs $2 --lr 0.001 --batch_size 2048 --dropout 0.5 > results/context/outputs_$1/UIC_graph_$1_NCF_epochs=$2.txt
echo "DONE NCF GCE-UIC"


#echo "Starting RANDOM CONTEXT EXPERIMENTS..."
#
#echo "Starting UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --random_context --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0 --not_early_stopping > results/context/outputs_$1/randomC_UIC_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UIC"
#python main.py --dataset $1 --algo_name fm --random_context --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/randomC_UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"
#
#
#echo "Starting RANDOM CONTEXT EXPERIMENTS with INIT WEIGHTS LOADED..."
#
#echo "Starting UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --load_init_weights --random_context --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UIC_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UIC"
#python main.py --dataset $1 --algo_name fm --load_init_weights --random_context --epochs $2 --lr 0.0005 --batch_size 2048 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"