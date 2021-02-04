#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


echo "Starting original EXTENDED CONTEXT FM experiments $1 ..."

echo "Starting FM..."
#
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 18744 > results/statistical_tests/$1_fm_reindexed_seed=18744_epochs=$2.txt
#echo "1/5 FM"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 00093 > results/statistical_tests/$1_fm_reindexed_seed=00093_epochs=$2.txt
#echo "2/5 FM"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 4232 > results/statistical_tests/$1_fm_reindexed_seed=4232_epochs=$2.txt
#echo "3/5  FM"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 11122 > results/statistical_tests/$1_fm_reindexed_seed=11122_epochs=$2.txt
#echo "4/5 FM"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 9898 > results/statistical_tests/$1_fm_reindexed_seed=9898_epochs=$2.txt
#echo "5/5 FM"
#echo "DONE FM REINDEXED
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 1234 > results/statistical_tests/$1_fm_reindexed_seed=1234_epochs=$2.txt
echo "1/5 FM"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 1111 > results/statistical_tests/$1_fm_reindexed_seed=1111_epochs=$2.txt
echo "2/5 FM"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 2222 > results/statistical_tests/$1_fm_reindexed_seed=2222_epochs=$2.txt
echo "3/5  FM"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 3333 > results/statistical_tests/$1_fm_reindexed_seed=3333_epochs=$2.txt
echo "4/5 FM"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 4444 > results/statistical_tests/$1_fm_reindexed_seed=4444_epochs=$2.txt
echo "5/5 FM"
#echo "DONE FM REINDEXED"
#
echo "Starting GRAPH FM..."
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 18744 > results/statistical_tests/$1_fm_graph_seed=18744_epochs=$2.txt
#echo "1/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 00093 > results/statistical_tests/$1_fm_graph_seed=00093_epochs=$2.txt
#echo "2/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 4232 > results/statistical_tests/$1_fm_graph_seed=4232_epochs=$2.txt
#echo "3/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 11122 > results/statistical_tests/$1_fm_graph_seed=11122_epochs=$2.txt
#echo "4/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 9898 > results/statistical_tests/$1_fm_graph_seed=9898_epochs=$2.txt
#echo "5/5  GRAPH"
 python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 1234 > results/statistical_tests/$1_fm_graph_seed=1234_epochs=$2.txt
echo "1/5  GRAPH"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 1111 > results/statistical_tests/$1_fm_graph_seed=1111_epochs=$2.txt
echo "2/5 GRAPH"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 2222 > results/statistical_tests/$1_fm_graph_seed=2222_epochs=$2.txt
echo "3/5 GRAPH"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 3333 > results/statistical_tests/$1_fm_graph_seed=3333_epochs=$2.txt
echo "4/5  GRAPH"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --seed 4444 > results/statistical_tests/$1_fm_graph_seed=4444_epochs=$2.txt
echo "5/5  GRAPH"
#echo "DONE GRAPH FM REINDEXED"
#
echo "Starting GRAPH FM WITH SIDE_INFORMATION..."
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 18744 > results/statistical_tests/$1_fm_SINFO_graph_seed=18744_epochs=$2.txt
#echo "1/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 00093 > results/statistical_tests/$1_fm_SINFO_graph_seed=00093_epochs=$2.txt
#echo "2/5  SINFO"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 4232 > results/statistical_tests/$1_fm_SINFO_graph_seed=4232_epochs=$2.txt
#echo "3/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 11122 > results/statistical_tests/$1_fm_SINFO_graph_seed=11122_epochs=$2.txt
#echo "4/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 9898 > results/statistical_tests/$1_fm_SINFO_graph_seed=9898_epochs=$2.txt
#echo "5/5 SINFO"
#echo "DONE GRAPH FM WITH SINFO"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 1234 > results/statistical_tests/$1_fm_SINFO_graph_seed=1234_epochs=$2.txt
echo "1/5 SINFO"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 1111 > results/statistical_tests/$1_fm_SINFO_graph_seed=1111_epochs=$2.txt
echo "2/5  SINFO"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 2222 > results/statistical_tests/$1_fm_SINFO_graph_seed=2222_epochs=$2.txt
echo "3/5 SINFO"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 3333 > results/statistical_tests/$1_fm_SINFO_graph_seed=3333_epochs=$2.txt
echo "4/5 SINFO"
python main.py --dataset $1 --rankall --algo_name fm --epochs $2 --lr 0.01 --batch_size 2048 --dropout 0 --gce --side_information --seed 4444 > results/statistical_tests/$1_fm_SINFO_graph_seed=4444_epochs=$2.txt
echo "5/5 SINFO"
echo "DONE GRAPH FM WITH SINFO"


echo "Starting original EXTENDED CONTEXT MF experiments $1 ..."

echo "Starting MF..."
#
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 18744 > results/statistical_tests/$1_mf_reindexed_seed=18744_epochs=$2.txt
#echo "1/5 MF"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 00093 > results/statistical_tests/$1_mf_reindexed_seed=00093_epochs=$2.txt
#echo "2/5 MF"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 4232 > results/statistical_tests/$1_mf_reindexed_seed=4232_epochs=$2.txt
#echo "3/5  MF"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 11122 > results/statistical_tests/$1_mf_reindexed_seed=11122_epochs=$2.txt
#echo "4/5 MF"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 9898 > results/statistical_tests/$1_mf_reindexed_seed=9898_epochs=$2.txt
#echo "5/5 MF"
 python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 1234 > results/statistical_tests/$1_mf_reindexed_seed=1234_epochs=$2.txt
echo "1/5 MF"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 1111 > results/statistical_tests/$1_mf_reindexed_seed=1111_epochs=$2.txt
echo "2/5 MF"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 2222 > results/statistical_tests/$1_mf_reindexed_seed=2222_epochs=$2.txt
echo "3/5  MF"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 3333 > results/statistical_tests/$1_mf_reindexed_seed=3333_epochs=$2.txt
echo "4/5 MF"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --seed 4444 > results/statistical_tests/$1_mf_reindexed_seed=4444_epochs=$2.txt
echo "5/5 MF"
echo "DONE MF REINDEXED"
#
echo "Starting GRAPH MF..."
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 18744 > results/statistical_tests/$1_mf_graph_seed=18744_epochs=$2.txt
#echo "1/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 00093 > results/statistical_tests/$1_mf_graph_seed=00093_epochs=$2.txt
#echo "2/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 4232 > results/statistical_tests/$1_mf_graph_seed=4232_epochs=$2.txt
#echo "3/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 11122 > results/statistical_tests/$1_mf_graph_seed=11122_epochs=$2.txt
#echo "4/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 9898 > results/statistical_tests/$1_mf_graph_seed=9898_epochs=$2.txt
#echo "5/5  GRAPH"
 python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 1234 > results/statistical_tests/$1_mf_graph_seed=1234_epochs=$2.txt
echo "1/5  GRAPH"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 1111 > results/statistical_tests/$1_mf_graph_seed=1111_epochs=$2.txt
echo "2/5 GRAPH"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 2222 > results/statistical_tests/$1_mf_graph_seed=2222_epochs=$2.txt
echo "3/5 GRAPH"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 3333 > results/statistical_tests/$1_mf_graph_seed=3333_epochs=$2.txt
echo "4/5  GRAPH"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce --seed 4444 > results/statistical_tests/$1_mf_graph_seed=4444_epochs=$2.txt
echo "5/5  GRAPH"
echo "DONE GRAPH MF REINDEXED"
#
echo "Starting GRAPH MF WITH SIDE_INFORMATION..."
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 18744 > results/statistical_tests/$1_mf_SINFO_graph_seed=18744_epochs=$2.txt
#echo "1/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 00093 > results/statistical_tests/$1_mf_SINFO_graph_seed=00093_epochs=$2.txt
#echo "2/5  SINFO"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 4232 > results/statistical_tests/$1_mf_SINFO_graph_seed=4232_epochs=$2.txt
#echo "3/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 11122 > results/statistical_tests/$1_mf_SINFO_graph_seed=11122_epochs=$2.txt
#echo "4/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 9898 > results/statistical_tests/$1_mf_SINFO_graph_seed=9898_epochs=$2.txt
#echo "5/5 SINFO"
#echo "DONE GRAPH MF WITH SINFO"
 python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 1234 > results/statistical_tests/$1_mf_SINFO_graph_seed=1234_epochs=$2.txt
echo "1/5 SINFO"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 1111 > results/statistical_tests/$1_mf_SINFO_graph_seed=1111_epochs=$2.txt
echo "2/5  SINFO"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 2222 > results/statistical_tests/$1_mf_SINFO_graph_seed=2222_epochs=$2.txt
echo "3/5 SINFO"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 3333 > results/statistical_tests/$1_mf_SINFO_graph_seed=3333_epochs=$2.txt
echo "4/5 SINFO"
python main.py --dataset $1 --rankall --algo_name mf --epochs $2 --lr 0.005 --batch_size 512 --dropout 0.5 --gce --side_information --seed 4444 > results/statistical_tests/$1_mf_SINFO_graph_seed=4444_epochs=$2.txt
echo "5/5 SINFO"
echo "DONE GRAPH MF WITH SINFO"
#


echo "Starting original EXTENDED CONTEXT NCF experiments $1 ..."

echo "Starting NCF..."

#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 18744 > results/statistical_tests/$1_mf_reindexed_seed=18744_epochs=$2.txt
#echo "1/5 MF"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 00093 > results/statistical_tests/$1_mf_reindexed_seed=00093_epochs=$2.txt
#echo "2/5 MF"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 4232 > results/statistical_tests/$1_mf_reindexed_seed=4232_epochs=$2.txt
#echo "3/5  MF"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 11122 > results/statistical_tests/$1_mf_reindexed_seed=11122_epochs=$2.txt
#echo "4/5 MF"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 9898 > results/statistical_tests/$1_mf_reindexed_seed=9898_epochs=$2.txt
#echo "5/5 MF"
python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 1234 > results/statistical_tests/$1_mf_reindexed_seed=1234_epochs=$2.txt
echo "1/5 MF"
python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 1111 > results/statistical_tests/$1_mf_reindexed_seed=1111_epochs=$2.txt
echo "2/5 MF"
python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 2222 > results/statistical_tests/$1_mf_reindexed_seed=2222_epochs=$2.txt
echo "3/5  MF"
python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 3333 > results/statistical_tests/$1_mf_reindexed_seed=3333_epochs=$2.txt
echo "4/5 MF"
python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --seed 4444 > results/statistical_tests/$1_mf_reindexed_seed=4444_epochs=$2.txt
echo "5/5 MF"
echo "DONE MF REINDEXED"
#


# TODO: tune
#echo "Starting GRAPH NCF..."
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 ?? --gce --seed 18744 > results/statistical_tests/$1_ncf_graph_seed=18744_epochs=$2.txt
#echo "1/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 ?? --gce --seed 00093 > results/statistical_tests/$1_ncf_graph_seed=00093_epochs=$2.txt
#echo "2/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 ?? --gce --seed 4232 > results/statistical_tests/$1_ncf_graph_seed=4232_epochs=$2.txt
#echo "3/5 GRAPH"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 ?? --gce --seed 11122 > results/statistical_tests/$1_ncf_graph_seed=11122_epochs=$2.txt
#echo "4/5  GRAPH"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 ?? --gce --seed 9898 > results/statistical_tests/$1_ncf_graph_seed=9898_epochs=$2.txt
#echo "5/5  GRAPH"
#echo "DONE GRAPH NCF REINDEXED"
##
#echo "Starting GRAPH NCF WITH SIDE_INFORMATION..."
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 18744 > results/statistical_tests/$1_ncf_SINFO_graph_seed=18744_epochs=$2.txt
#echo "1/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 00093 > results/statistical_tests/$1_ncf_SINFO_graph_seed=00093_epochs=$2.txt
#echo "2/5  SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 4232 > results/statistical_tests/$1_ncf_SINFO_graph_seed=4232_epochs=$2.txt
#echo "3/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 11122 > results/statistical_tests/$1_ncf_SINFO_graph_seed=11122_epochs=$2.txt
#echo "4/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 9898 > results/statistical_tests/$1_ncf_SINFO_graph_seed=9898_epochs=$2.txt
#echo "5/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 1234 > results/statistical_tests/$1_ncf_SINFO_graph_seed=1234_epochs=$2.txt
#echo "1/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 1111 > results/statistical_tests/$1_ncf_SINFO_graph_seed=1111_epochs=$2.txt
#echo "2/5  SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 2222 > results/statistical_tests/$1_ncf_SINFO_graph_seed=2222_epochs=$2.txt
#echo "3/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 3333 > results/statistical_tests/$1_ncf_SINFO_graph_seed=3333_epochs=$2.txt
#echo "4/5 SINFO"
#python main.py --dataset $1 --rankall --algo_name ncf --epochs $2 --lr 0.001 --batch_size 1024 --dropout 0.5 --gce --side_information --seed 4444 > results/statistical_tests/$1_ncf_SINFO_graph_seed=4444_epochs=$2.txt
echo "5/5 SINFO"
#echo "DONE GRAPH NCF WITH SINFO"
