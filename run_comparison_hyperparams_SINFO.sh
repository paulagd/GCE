#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name


echo "Starting tunning SINFO parameters experiments in dataset $1 ..."

python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --side_information
echo "DONE MF TUNNING - GCE - SINFO"
python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --side_information
echo "DONE FM TUNNING - GCE- SINFO "
#python tune.py --dataset $1 --algo_name nfm --not_early_stopping --gce --side_information
#echo "DONE NFM TUNNING - GCE - SINFO"
#python tune.py --dataset $1 --algo_name deepfm --not_early_stopping --gce --side_information
#echo "DONE DFM TUNNING - GCE - SINFO"
#python tune.py --dataset $1 --algo_name ncf --not_early_stopping --gce --side_information
#echo "DONE NCF TUNNING - GCE - SINFO"

