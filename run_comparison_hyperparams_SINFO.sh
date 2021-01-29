#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name


echo "Starting tunning SINFO parameters experiments in dataset $1 ..."

#echo "Starting UII EXPERIMENTS..."
#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --side_information --uii --tune_epochs 50
#echo "DONE MF TUNNING - GCE - SINFO - UII" #(done)
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --side_information --uii --tune_epochs 50
#echo "DONE FM TUNNING - GCE- SINFO - UII" #(doing)

echo "Starting UIC EXPERIMENTS..."
#python tune.py --dataset $1 --algo_name mf --not_early_stopping --gce --side_information --actors --tune_epochs 50
#echo "DONE MF TUNNING - GCE - SINFO- UIC --actors "
python tune.py --dataset $1 --algo_name $2 --not_early_stopping --gce --side_information --tune_epochs 50
echo "DONE $2 TUNNING - GCE- SINFO - UIC "  #( doing aux2)

#python tune.py --dataset $1 --algo_name nfm --not_early_stopping --gce --side_information
#echo "DONE NFM TUNNING - GCE - SINFO"
#python tune.py --dataset $1 --algo_name deepfm --not_early_stopping --gce --side_information
#echo "DONE DFM TUNNING - GCE - SINFO"
#python tune.py --dataset $1 --algo_name ncf --not_early_stopping --gce --side_information
#echo "DONE NCF TUNNING - GCE - SINFO"

#echo "Starting SINFO DRUG AND DISEASE..."
#
#python tune.py --dataset $1 --algo_name fm --not_early_stopping --gce --side_information --context_as_userfeat
