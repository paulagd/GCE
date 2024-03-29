#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##params --> dataset_name model


echo "Starting NO CONTEXT tunning parameters experiments in dataset $1 ..."
##
python tune.py --dataset $1 --algo_name mf --context --tune_epochs 50
echo "DONE MF TUNNING"
python tune.py --dataset $1 --algo_name fm --context --tune_epochs 50
echo "DONE FM TUNNING"
###
##
#echo "Starting tunning parameters experiments in dataset $1 - $2..."
#echo "Starting UIC EXPERIMENTS..."
#
#python tune.py --dataset $1 --algo_name $2 --not_early_stopping --tune_epochs 50
#echo "DONE $2 TUNNING"

echo "Starting GRAPH tunning parameters experiments in dataset $1 ..."

echo "Starting UIC EXPERIMENTS..."

python tune.py --dataset $1 --algo_name mf --gce --tune_epochs 50 --context
echo "DONE MF TUNNING - GCE"
python tune.py --dataset $1 --algo_name fm --gce --tune_epochs 50 --context
echo "DONE FM TUNNING - GCE"

#python tune.py --dataset $1 --algo_name $2 --not_early_stopping --gce --gcetype sgc --mh 2 --tune_epochs 50
#echo "DONE $2 TUNNING - GCE"
#
#python tune.py --dataset $1 --algo_name $2 --not_early_stopping --gce --gcetype sage --tune_epochs 50
#echo "DONE $2 TUNNING - GCE"

#echo "Starting SINFO DRUG AND DISEASE..."
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --side_information --context_as_userfeat --tune_epochs 50
