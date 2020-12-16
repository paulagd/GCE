#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original no context experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name ncf --context --epochs $2 > results/no_context/outputs_$1/reindexed_$1_ncf_epochs=$2.txt
echo "DONE NCF REINDEXED"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --algo_name mf --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name ncf --gce --context --epochs $2 > results/no_context/outputs_$1/graph_$1_ncf_epochs=$2.txt
echo "DONE NCF REINDEXED-GCE"


echo "Starting original CONTEXT experiments $1 ..."
echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 > results/context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --epochs $2 > results/context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name ncf --epochs $2 > results/context/outputs_$1/reindexed_$1_ncf_epochs=$2.txt
echo "DONE NCF ORIGINAL"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --algo_name mf --gce --epochs $2 > results/context/outputs_$1/graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce --epochs $2 > results/context/outputs_$1/graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
python main.py --dataset $1 --algo_name ncf --gce --epochs $2 > results/context/outputs_$1/graph_$1_ncf_epochs=$2.txt
echo "DONE NCF REINDEXED-GCE"