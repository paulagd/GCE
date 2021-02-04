#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

echo "Starting QUANTILES  CONTEXT experiments on $3 in $1 ..."
python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_reindexed_$1_fm.txt
python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping > results/context/outputs_$1/QUANTILE=$2_$3_graph_$1_fm.txt

#echo "Starting QUANTILES STATISTICS EXPERIMENTS..."
#
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping --seed 18744 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_reindexed_$1_fm_seed=18744.txt
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --seed 18744 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_graph_$1_fm_seed=18744.txt
#echo "1/5 MF"
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping --seed 00093 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_reindexed_$1_fm_seed=00093.txt
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --seed 00093 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_graph_$1_fm_seed=00093.txt
#echo "2/5 MF"
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping --seed 4232 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_reindexed_$1_fm_seed=4232.txt
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --seed 4232 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_graph_$1_fm_seed=4232.txt
#echo "3/5 MF"
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping --seed 11122 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_reindexed_$1_fm_seed=11122.txt
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --seed 11122 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_graph_$1_fm_seed=11122.txt
#echo "4/5 MF"
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping --seed 9898 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_reindexed_$1_fm_seed=9898.txt
#python main.py --dataset $1 --rankall --algo_name fm --remove_top_users $2 --remove_on $3 --epochs 100 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --not_early_stopping --seed 9898 > results/statistical_tests/QUANTILES/QUANTILE=$2_$3_graph_$1_fm_seed=9898.txt
#echo "5/5 MF"






