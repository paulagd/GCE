#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs lr dropout


echo "Starting tunning parameters experiments in dataset $1 ..."
echo "Starting UIC EXPERIMENTS..."

python tune.py --dataset $1 --algo_name fm --context_type weekday --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING WEEKDAY"
python tune.py --dataset $1 --algo_name fm --context_type isweekend --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING ISWEEKEND"
python tune.py --dataset $1 --algo_name fm --context_type homework --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING HOMEWORK"
python tune.py --dataset $1 --algo_name fm --context_type cost --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING COST"
python tune.py --dataset $1 --algo_name fm --context_type weather --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING WEATHER"
python tune.py --dataset $1 --algo_name fm --context_type country --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING COUNTRY"
python tune.py --dataset $1 --algo_name fm --context_type city --not_early_stopping --tune_epochs 50
echo "DONE FM TUNNING CITY"