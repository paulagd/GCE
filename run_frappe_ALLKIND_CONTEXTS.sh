#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs

echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting CONTEXT EXPERIMENTS..."

#python main.py --dataset $1 --algo_name fm --random_context --epochs $2 --lr 0.001 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/INIT_randomC_UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"

#echo "Starting WEEKDAY EXPERIMENTS..."
#python main.py --dataset $1 --algo_name fm --epochs $2 --context_type weekday --lr 0.01 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"  #done
#python main.py --dataset $1 --algo_name fm --gce --epochs $2 --context_type ? --lr 0. --batch_size  --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM graph-UIC"  #done


echo "Starting isweekend EXPERIMENTS..."
python main.py --dataset $1 --algo_name fm --epochs $2 --context_type isweekend --lr 0.0005 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"  #done

echo "Starting weather EXPERIMENTS..."
python main.py --dataset $1 --algo_name fm --epochs $2 --context_type weather --lr 0.0005 --batch_size 2048 --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"  #done

echo "Starting homework EXPERIMENTS..."
python main.py --dataset $1 --algo_name fm --epochs $2 --context_type homework --lr 0.005 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"  #done

#echo "Starting country EXPERIMENTS..."
#python main.py --dataset $1 --algo_name fm --epochs $2 --context_type country --lr 0.01 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"  #done
#
#echo "Starting cost EXPERIMENTS..."
#python main.py --dataset $1 --algo_name fm --epochs $2 --context_type cost --lr 0.0001 --batch_size 512 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"  #done
#
#echo "Starting CITY EXPERIMENTS..."
#python main.py --dataset $1 --algo_name fm --epochs $2 --context_type city --lr 0.001 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
#echo "DONE FM REINDEXED-UIC"  #done


