#!/bin/bash


echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf --epochs $2 > results/context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm --epochs $2 > results/context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"
python main.py --dataset $1 --algo_name nfm --epochs $2 > results/context/outputs_$1/reindexed_$1_nfm_epochs=$2.txt
echo "DONE NFM ORIGINAL"
python main.py --dataset $1 --algo_name deepfm --epochs $2 > results/context/outputs_$1/reindexed_$1_deepfm_epochs=$2.txt
echo "DONE DFM ORIGINAL"


#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
#MF :      --prepro 10filter        --num_ng 8        --factors 92       --lr 0.0053
#FM :      --prepro 10filter        --num_ng 9        --factors 61       --lr 0.0031


#python main.py --dataset $1 --algo_name mf --reindex --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/original/recommended_$1_mf.txt
#echo "DONE MF ORIGINAL"
#python main.py --dataset $1 --algo_name fm --reindex --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/original/recommended_$1_fm.txt
#echo "DONE FM ORIGINAL"
#python main.py --dataset $1 --algo_name neumf --reindex > outputs/original/$1_neufm.txt
#echo "DONE NEUMF ORIGINAL"
#python main.py --dataset $1 --algo_name nfm --reindex > outputs/original/$1_nfm.txt
#echo "DONE NFM ORIGINAL"

#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/reindexed/recommended_$1_mf.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset $1 --algo_name fm --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/reindexed/recommended_$1_fm.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf > outputs/reindexed/neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm > outputs/reindexed/$1_nfm.txt
#echo "DONE NFM REINDEXED"

#echo "Starting GRAPH experiments..."
#python main.py --dataset $1 --algo_name mf --gce --prepro 10filter --num_ng 2 --factors 34 --lr 0.0005 > outputs/graph/recommended_$1_mf.txt
#echo "DONE MF REINDEXED-GCE"
#python main.py --dataset $1 --algo_name fm --gce --prepro 10filter --num_ng 6 --factors 78 --lr 0.0015 > outputs/graph/recommended_$1_fm.txt
#echo "DONE FM REINDEXED-GCE"
#python main.py --dataset ml-100k --algo_name neumf --gce > outputs/graph/$1_neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset $1 --algo_name nfm  --gce > outputs/graph/$1_nfm.txt
#echo "DONE NFM REINDEXED-GCE"