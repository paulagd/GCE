#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --rankall > results/$1/rankall_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --rankall > results/$1/rankall_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/reindexed_$1_ncf_epochs=$2.txt
#echo "DONE NCF REINDEXED"

echo "Starting SGC GRAPH EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --gcetype sgc --rankall > results/$1/rankall_SGChop1_graph_$1_mf_epochs=$2.txt
echo "DONE MF-GCE"
python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --gcetype sgc --rankall > results/$1/rankall_SGChop1_graph_$1_fm_epochs=$2.txt
echo "DONE FM-GCE"


#echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --rankall > results/$1/rankall_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE"
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --rankall > results/$1/rankall_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_ncf_epochs=$2.txt
#echo "DONE NCF-GCE" #to do
#
#echo "Starting GRAPH EXPERIMENTS ATTENTION..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --num_heads 10 --lr 0.01 --batch_size 512 --dropout 0.5 --gce --attention --rankall > results/$1/rankall_GAT_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE"
#python main.py --dataset $1 --algo_name fm --epochs $2 --num_heads 3 --lr 0.01 --batch_size 2048 --dropout 0.5 --gce --attention --rankall > results/$1/rankall_GAT_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/graph_$1_ncf_epochs=$2.txt
#echo "DONE NCF-GCE" #to do
#
#echo "Starting GRAPH EXPERIMENTS WITH GENDER as SIDE INFO ON X MATRIX..."
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"   # doing script
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-gender SIDE-INFO"  # doing script
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_ncf_epochs=$2.txt
#echo "DONE NCF-GCE-GENDR SIDE-INFO"