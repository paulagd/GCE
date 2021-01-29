#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##params --> dataset_name model


#echo "Starting NO CONTEXT tunning parameters experiments in dataset $1 ..."
##
#python tune.py --dataset $1 --algo_name mf --context --not_early_stopping --tune_epochs 50
#echo "DONE MF TUNNING"
#python tune.py --dataset $1 --algo_name fm --context --not_early_stopping --tune_epochs 50
#echo "DONE FM TUNNING"
##
##
echo "Starting tunning parameters experiments in dataset $1 - $2..."
echo "Starting UIC EXPERIMENTS..."

python tune.py --dataset $1 --algo_name $2 --not_early_stopping --tune_epochs 50
echo "DONE $2 TUNNING"

echo "Starting GRAPH tunning parameters experiments in dataset $1 ..."

echo "Starting UIC EXPERIMENTS..."
#
#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --tune_epochs 50
#echo "DONE MF TUNNING - GCE"
python tune.py --dataset $1 --algo_name $2 --not_early_stopping --gce --tune_epochs 50
echo "DONE $2 TUNNING - GCE"


echo "Starting tunning SINFO parameters experiments in dataset $1 ..."

python tune.py --dataset $1 --algo_name $2 --not_early_stopping --gce --side_information --tune_epochs 50
echo "DONE $2 TUNNING - GCE- SINFO - UIC "

#echo "Starting Multi hop 2 -- GRAPH - UIC EXPERIMENTS..."

#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --tune_epochs 50
#echo "DONE MF TUNNING - GCE"
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --mh 2 --tune_epochs 50
#echo "DONE FM TUNNING - GCE"

#echo "Starting SINFO DRUG AND DISEASE..."
#
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --side_information --context_as_userfeat --tune_epochs 50
