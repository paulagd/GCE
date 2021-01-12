#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs lr dropout


#echo "Starting tunning parameters experiments in dataset $1 ..."

#python tune.py --dataset $1 --algo_name mf --context --not_early_stopping --tune_epochs 50
#echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --context --not_early_stopping --tune_epochs 50
#echo "DONE FM TUNNING"


#echo "Starting tunning parameters experiments in dataset $1 ..."
#
#echo "Starting UII EXPERIMENTS..."
#
#python tune.py --dataset $1 --algo_name mf --uii --not_early_stopping --tune_epochs 50
#echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --uii --not_early_stopping --tune_epochs 50
#echo "DONE FM TUNNING"
#
#echo "Starting UIC EXPERIMENTS..."
#
#python tune.py --dataset $1 --algo_name mf --not_early_stopping --tune_epochs 50
#echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --tune_epochs 50
#echo "DONE FM TUNNING"
############################################

#python tune.py --dataset $1 --algo_name mf --not_early_stopping
#echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping
#echo "DONE FM TUNNING"
#python tune.py --dataset $1 --algo_name nfm --not_early_stopping
#echo "DONE NFM TUNNING"
#python tune.py --dataset $1 --algo_name deepfm --not_early_stopping
#echo "DONE DFM TUNNING"
#python tune.py --dataset $1 --algo_name ncf --not_early_stopping
#echo "DONE NCF TUNNING"
############################################


echo "Starting GRAPH tunning parameters experiments in dataset $1 ..."
echo "Starting UII EXPERIMENTS..."

python tune.py --dataset $1 --algo_name mf --not_early_stopping --uii --gce --tune_epochs 50
echo "DONE MF TUNNING - GCE"
python tune.py --dataset $1 --algo_name fm --not_early_stopping --uii --gce --tune_epochs 50
echo "DONE FM TUNNING - GCE"

echo "Starting UIC EXPERIMENTS..."

python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --tune_epochs 50
echo "DONE MF TUNNING - GCE"
python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --tune_epochs 50
echo "DONE FM TUNNING - GCE"

#python tune.py --dataset $1 --algo_name nfm --not_early_stopping --gce
#echo "DONE NFM TUNNING - GCE"
#python tune.py --dataset $1 --algo_name deepfm --not_early_stopping --gce
#echo "DONE DFM TUNNING - GCE"
#python tune.py --dataset $1 --algo_name ncf --not_early_stopping --gce
#echo "DONE NCF TUNNING - GCE"

