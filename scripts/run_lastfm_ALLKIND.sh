#!/bin/bash

#CUDA_VISIBLE_DEVICES=2
##sh run_models_script.sh dataset_name epochs

echo "Starting original EXTENDED CONTEXT experiments $1 ..."

#echo "Starting NO CONTEXT EXPERIMENTS..."
#python main.py --dataset $1 --algo_name fm --rankall --context --epochs $2 --lr 0.001 --batch_size 256 --dropout 0 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED"  #done
#
echo "Starting CONTEXT EXPERIMENTS..."

python main.py --dataset $1 --algo_name mf --rankall --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 > results/context/outputs_$1/UIC_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --rankall --epochs $2 --lr 0.001 --batch_size 256 --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"
python main.py --dataset $1 --algo_name ncf --rankall --epochs $2 --lr 0.0001 --batch_size 512 --dropout 0.15 > results/context/outputs_$1/UIC_reindexed_$1_NCF_epochs=$2.txt
echo "DONE NCF REINDEXED-UIC"
                                
echo "Starting GRAPH EXPERIMENTS..."
#
python main.py --dataset $1 --algo_name mf --gce --rankall --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.15  --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --gce --rankall --epochs $2 --lr 0.005 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
echo "DONE FM GCE-UIC"
python main.py --dataset $1 --algo_name ncf --gce --rankall --epochs $2  --lr 0.001 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_NCF_epochs=$2.txt
echo "DONE NCF REINDEXED-UIC"

echo "Starting GRAPH EXPERIMENTS WITH ARTIST SIDE INFO ON X MATRIX and CONTEXT NODES..."

python main.py --dataset $1 --algo_name mf --rankall --epochs $2 --lr 0.005 --batch_size 512 --dropout 0  --gce --side_information > results/context/outputs_$1/SI_UIC_graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-UIC"
python main.py --dataset $1 --algo_name fm --rankall --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0.5 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_fm_epochs=$2.txt
echo "DONE FM-GCE-GENDR SIDE-INFO"
python main.py --dataset $1 --algo_name ncf --rankall --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.15 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_ncf_epochs=$2.txt
echo "DONE NCF REINDEXED-UIC"
