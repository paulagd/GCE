#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs


#echo "Starting original EXTENDED CONTEXT experiments ML-100K ..."

#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset ml-100k --rankall --algo_name mf --epochs $1 --lr 0.0005 --batch_size 512 --dropout 0 > results/convergence/ml-100k_reindexed_mf_epochs=$1.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset ml-100k --rankall --algo_name fm --epochs $1 --lr 0.0005 --batch_size 512 --dropout 0 > results/convergence/ml-100k_reindexed_fm_epochs=$1.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset ml-100k --rankall --algo_name ncf --epochs $1 --lr 0.001 --batch_size 1024 --dropout 0.5 > results/convergence/ml-100k_reindexed_ncf_epochs=$1.txt
#echo "DONE NCF REINDEXED"
#
#echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset ml-100k --rankall --algo_name mf --epochs $1 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce > results/convergence/ml-100k_graph_ml-100k_mf_epochs=$1.txt
#echo "DONE MF-GCE"
#python main.py --dataset ml-100k --rankall --algo_name fm --epochs $1 --lr 0.005 --batch_size 512 --dropout 0.5 --gce > results/convergence/ml-100k_graph_fm_epochs=$1.txt
#echo "DONE FM-GCE"
#python main.py --dataset ml-100k --rankall --algo_name ncf --epochs $1 --lr 0.0001 --batch_size 256 --dropout 0 --gce > results/convergence/ml-100k_graph_ml-100k_ncf_epochs=$1.txt
#echo "DONE NCF-GCE"


#echo "Starting original EXTENDED CONTEXT experiments BOOKS ..."
#
#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset books --rankall --algo_name mf --epochs $1 --lr 0.0005 --batch_size 256 --dropout 0.15 > results/convergence/books_reindexed_mf_epochs=$1.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset books --rankall --algo_name fm --epochs $1 --lr 0.0005 --batch_size 2048 --dropout 0.15 > results/convergence/books_reindexed_fm_epochs=$1.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset books --rankall --algo_name ncf --epochs $1 --lr 0.0001 --batch_size 512 --dropout 0.5 > results/convergence/books_reindexed_ncf_epochs=$1.txt
#echo "DONE NCF REINDEXED"
#
#echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset books --rankall --algo_name mf --epochs $1 --lr 0.005 --batch_size 2048 --dropout 0.5 --gce > results/convergence/books_graph_mf_epochs=$1.txt
#echo "DONE MF-GCE"
#python main.py --dataset books --rankall --algo_name fm --epochs $1 --lr 0.01 --batch_size 2048 --dropout 0 --gce > results/convergence/books_graph_fm_epochs=$1.txt
#echo "DONE FM-GCE"
##python main.py --dataset books --rankall --algo_name ncf --epochs $1 --lr 0.01 --batch_size 2048 --dropout 0 --gce > results/cconvergence/books_graph_ncf_epochs=$1.txt
##echo "DONE NCF-GCE"
#
#
#echo "Starting original EXTENDED CONTEXT experiments DRUGS ..."
#
#echo "Starting REINDEXED EXPERIMENTS..."
#python main.py --dataset drugs --rankall --algo_name mf --epochs $1 --lr 0.0001 --batch_size 2048 --dropout 0.15 > results/convergence/drugs_reindexed_mf_epochs=$1.txt
#echo "DONE MF REINDEXED"
#python main.py --dataset drugs --rankall --algo_name fm --epochs $1 --lr 0.001 --batch_size 2048 --dropout 0.15 > results/convergence/drugs_reindexed_fm_epochs=$1.txt
#echo "DONE FM REINDEXED"
#python main.py --dataset drugs --rankall --algo_name ncf --epochs $1 --lr 0.001 --batch_size 512 --dropout 0.5 > results/convergence/drugs_reindexed_ncf_epochs=$1.txt
#echo "DONE NCF REINDEXED"
#
#echo "Starting GRAPH EXPERIMENTS..."
#python main.py --dataset drugs --rankall --algo_name mf --epochs $1 --lr 0.0001 --batch_size 2048 --dropout 0.5 --gce > results/convergence/drugs_graph_mf_epochs=$1.txt
#echo "DONE MF-GCE"
#python main.py --dataset drugs --rankall --algo_name fm --epochs $1 --lr 0.01 --batch_size 2048 --dropout 0 --gce > results/convergence/drugs_graph_fm_epochs=$1.txt
#echo "DONE FM-GCE"
#python main.py --dataset drugs --rankall --algo_name ncf --epochs $1 --lr 0.0001 --batch_size 512 --dropout 0.5 --gce > results/convergence/drugs_graph_ncf_epochs=$1.txt
#echo "DONE NCF-GCE"