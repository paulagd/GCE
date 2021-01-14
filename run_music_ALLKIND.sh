#!/bin/bash

#CUDA_VISIBLE_DEVICES=2
##sh run_models_script.sh dataset_name epochs

echo "Starting original EXTENDED CONTEXT experiments $1 ..."

#echo "Starting NO CONTEXT EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --context --epochs $2 --lr 0.01 --batch_size 512 --dropout 0 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED" #done
#python main.py --dataset $1 --algo_name fm --context --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED"  #done
#
#
#echo "Starting CONTEXT EXPERIMENTS..."
#
#echo "Starting UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --save_initial_weights --uii --epochs $2 --lr 0.005 --batch_size 256 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UII" #done
#python main.py --dataset $1 --algo_name fm --save_initial_weights --uii --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UII"  #done
#
#echo "Starting UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --save_initial_weights --epochs $2 --lr 0.01 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UIC" #done
#python main.py --dataset $1 --algo_name fm --save_initial_weights --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"  #done
##
echo "Starting GRAPH EXPERIMENTS..."
echo "Starting GCE-UII EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --save_initial_weights --uii --gce --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_mf_epochs=$2.txt
echo "DONE MF GCE-UII" #done
-, -, -, 4, 64, 0.5,+ 0.01, 2048, 0, 0.01
python main.py --dataset $1 --algo_name fm --save_initial_weights --uii --gce --epochs $2 --lr 0.005 --batch_size 256 --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_fm_epochs=$2.txt
echo "DONE FM GCE-UII"  # doing script

echo "Starting GCE-UIC EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --save_initial_weights --gce --epochs $2 --lr 0.01 --batch_size 256 --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_mf_epochs=$2.txt
echo "DONE MF GCE-UIC"  # doing script
python main.py --dataset $1 --algo_name fm --save_initial_weights --gce --epochs $2 --lr 0.01 --batch_size 256 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
echo "DONE FM GCE-UIC"  # doing script

#echo "Starting GRAPH EXPERIMENTS WITH GENDER SIDE INFO ON X MATRIX..."
#echo "Starting GCE-UII EXPERIMENTS..."
#
#python main.py --dataset $1 --algo_name mf --uii --epochs $2 --lr ? --batch_size ? --dropout ? --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UII_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"   #done
#python main.py --dataset $1 --algo_name fm --uii --epochs $2 --lr ? --batch_size ? --dropout ? --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UII_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO"  # doing script
#
#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr ? --batch_size ? --dropout ? --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"   # doing script
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr ? --batch_size ? --dropout ? --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO"  # doing script