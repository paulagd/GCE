#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_baselines.sh dataset_name


echo "Starting original experiments ..."

python main.py --dataset $1 --algo_name mf --reindex > outputs/original/$1_mf.txt
echo "DONE MF ORIGINAL"
python main.py --dataset $1 --algo_name fm --reindex > outputs/original/$1_fm.txt
echo "DONE FM ORIGINAL"
#python main.py --dataset $1 --algo_name neumf --reindex > outputs/original/$1_neufm.txt
#echo "DONE NEUMF ORIGINAL"
python main.py --dataset $1 --algo_name nfm --reindex > outputs/original/$1_nfm.txt
echo "DONE NFM ORIGINAL"

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset $1 --algo_name mf > outputs/reindexed/$1_mf.txt
echo "DONE MF REINDEXED"
python main.py --dataset $1 --algo_name fm > outputs/reindexed/$1_fm.txt
echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf > outputs/reindexed/neumf.txt
#echo "DONE NEUFM REINDEXED"
python main.py --dataset $1 --algo_name nfm > outputs/reindexed/$1_nfm.txt
echo "DONE NFM REINDEXED"

echo "Starting GRAPH experiments..."
python main.py --dataset $1 --algo_name mf --gce > outputs/graph/$1_mf.txt
echo "DONE MF REINDEXED-GCE"
python main.py --dataset $1 --algo_name fm --gce > outputs/graph/$1_fm.txt
echo "DONE FM REINDEXED-GCE"
#python main.py --dataset ml-100k --algo_name neumf --gce > outputs/graph/$1_neumf.txt
#echo "DONE NEUFM REINDEXED"
python main.py --dataset $1 --algo_name nfm  --gce > outputs/graph/$1_nfm.txt
echo "DONE NFM REINDEXED-GCE"

#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
