#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs lr dropout


echo "Starting NO CONTEXT tunning parameters experiments in dataset $1 ..."
##
#python tune.py --dataset $1 --algo_name mf --context --not_early_stopping --tune_epochs 50
#echo "DONE MF TUNNING"
python tune.py --dataset $1 --algo_name fm --context --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING"
##
##
echo "Starting tunning parameters experiments in dataset $1 ..."
##
##echo "Starting UII EXPERIMENTS..."
##
##python tune.py --dataset $1 --algo_name mf --uii --not_early_stopping --tune_epochs 50
##echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --uii --not_early_stopping --tune_epochs 50
#echo "DONE FM TUNNING"
##
echo "Starting UIC EXPERIMENTS..."

#python tune.py --dataset $1 --algo_name mf --not_early_stopping --tune_epochs 50
##echo "DONE MF TUNNING"
python tune.py --dataset $1 --algo_name fm --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING"
##
echo "Starting GRAPH tunning parameters experiments in dataset $1 ..."
##echo "Starting UII EXPERIMENTS..."
#
##python tune.py --dataset $1 --algo_name mf --not_early_stopping --uii --gce --tune_epochs 50
##echo "DONE MF TUNNING - GCE"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --uii --gce --tune_epochs 50
#echo "DONE FM TUNNING - GCE"
#
echo "Starting UIC EXPERIMENTS..."
#
#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --tune_epochs 50
#echo "DONE MF TUNNING - GCE"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --tune_epochs 50
#echo "DONE FM TUNNING - GCE"

echo "Starting Multi hop 2 -- GRAPH - UIC EXPERIMENTS..."

#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --tune_epochs 50
#echo "DONE MF TUNNING - GCE"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --mh 2 --tune_epochs 50
#echo "DONE FM TUNNING - GCE"

