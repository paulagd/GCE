#!/bin/bash

#CUDA_VISIBLE_DEVICES=1

echo "Starting original experiments ..."

python main.py --dataset ml-100k --algo_name mf --reindex > outputs/original/mf.txt
echo "DONE MF ORIGINAL"
python main.py --dataset ml-100k --algo_name fm --reindex > outputs/original/fm.txt
echo "DONE FM ORIGINAL"
python main.py --dataset ml-100k --algo_name neumf --reindex > outputs/original/neufm.txt
echo "DONE NEUMF ORIGINAL"
python main.py --dataset ml-100k --algo_name nfm --reindex > outputs/original/nfm.txt
echo "DONE NFM ORIGINAL"

echo "Starting REINDEXED EXPERIMENTS..."
python main.py --dataset ml-100k --algo_name mf > outputs/reindexed/mf.txt
echo "DONE MF REINDEXED"
python main.py --dataset ml-100k --algo_name fm > outputs/reindexed/fm.txt
echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf > outputs/reindexed/neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset ml-100k --algo_name nfm > outputs/reindexed/nfm.txt
#echo "DONE NFM REINDEXED"

echo "Starting GRAPH experiments..."
python main.py --dataset ml-100k --algo_name mf --gce > outputs/graph/mf.txt
echo "DONE MF REINDEXED"
python main.py --dataset ml-100k --algo_name fm --gce > outputs/graph/fm.txt
echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --algo_name neumf --gce > outputs/graph/neumf.txt
#echo "DONE NEUFM REINDEXED"
#python main.py --dataset ml-100k --algo_name nfm  --gce > outputs/graph/nfm.txt
#echo "DONE NFM REINDEXED"

#echo "Starting GRAPH experiments ML MULTIHOP 2 ..."
