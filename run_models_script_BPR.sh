#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --context --epochs $2 > results/no_context/outputs_$1/BPR_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --context --epochs $2 > results/no_context/outputs_$1/BPR_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED"
##python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --context --mf --epochs $2 > results/no_context/outputs_$1/BPR_reindexed_$1_nfm_epochs=$2.txt
##echo "DONE NFM ORIGINAL"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --context --epochs $2 > results/no_context/outputs_$1/BPR_reindexed_$1_nfm_pairwise_epochs=$2.txt
#echo "DONE NFM PAIRWISE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name deepfm --context --epochs $2 > results/no_context/outputs_$1/BPR_eindexed_$1_deepfm_epochs=$2.txt
#echo "DONE DFM ORIGINAL"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name ncf --context --epochs $2 > results/no_context/outputs_$1/BPR_reindexed_$1_ncf_epochs=$2.txt
#echo "DONE NCF ORIGINAL"
#
#echo "Starting GRAPH experiments..."
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-GCE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-GCE"
##python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --mf --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_nfm_epochs=$2.txt
##echo "DONE NFM REINDEXED-GCE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_nfm_pairwise_epochs=$2.txt
#echo "DONE NFM REINDEXED-GCE-PAIRWISE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name deepfm --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_deepfm_epochs=$2.txt
#echo "DONE DFM REINDEXED-GCE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name ncf --gce --context --epochs $2 > results/no_context/outputs_$1/BPR_graph_$1_ncf_epochs=$2.txt
#echo "DONE NCF REINDEXED-GCE"


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --mf --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_nfm_epochs=$2.txt
#echo "DONE NFM ORIGINAL"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_nfm_pairwise_epochs=$2.txt
echo "DONE NFM - PAIRWISE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name deepfm --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_deepfm_epochs=$2.txt
echo "DONE DFM ORIGINAL"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name ncf --epochs $2 > results/context/outputs_$1/BPR_reindexed_$1_ncf_epochs=$2.txt
echo "DONE NCF ORIGINAL"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name mf --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name fm --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-GCE"
#python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --mf --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_nfm_epochs=$2.txt
#echo "DONE NFM REINDEXED-GCE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name nfm --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_nfm_pairwise_epochs=$2.txt
echo "DONE NFM REINDEXED-GCE-PAIRWISE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name deepfm --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_deepfm_epochs=$2.txt
echo "DONE DFM REINDEXED-GCE"
python main.py --dataset $1 --problem_type pair --loss BPR --algo_name ncf --gce --epochs $2 > results/context/outputs_$1/BPR_graph_$1_ncf_epochs=$2.txt
echo "DONE NCF REINDEXED-GCE"
