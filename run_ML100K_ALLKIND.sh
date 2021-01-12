#!/bin/bash

#CUDA_VISIBLE_DEVICES=1
##sh run_models_script.sh dataset_name epochs
s
echo "Starting original EXTENDED CONTEXT experiments $1 ..."

echo "Starting NO CONTEXT EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --context --epochs $2 --lr 0.001 --batch_size 512 --dropout 0 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED" #done
python main.py --dataset $1 --algo_name fm --context --epochs $2 --lr 0.001 --batch_size 2048 --dropout 0.5 --not_early_stopping > results/no_context/outputs_$1/reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED"  #done

echo "Starting UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --uii --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UII" #done
python main.py --dataset $1 --algo_name fm --uii --epochs $2 --lr 0.0005 --batch_size 1024 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/UII-reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UII"  #done

echo "Starting UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr 0.001 --batch_size 512 --dropout 0.15 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_mf_epochs=$2.txt
#echo "DONE MF REINDEXED-UIC" #done
python main.py --dataset $1 --algo_name fm --epochs $2 --lr 0.0005 --batch_size 512 --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_reindexed_$1_fm_epochs=$2.txt
echo "DONE FM REINDEXED-UIC"  #done


#echo "Starting GRAPH EXPERIMENTS..."
#
#echo "Starting GCE-UII EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --uii --gce --epochs $2 --lr 0.005 --batch_size 256 --dropout 0.5 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UII" #done

#python main.py --dataset $1 --algo_name fm --uii --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UII-graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UII"  (tunning croissant)
#
#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF GCE-UIC"
#python main.py --dataset $1 --algo_name fm --gce --epochs $2 --lr ? --batch_size ? --dropout 0 --not_early_stopping > results/context/outputs_$1/UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM GCE-UIC" (tunning alone crepe)


#echo "Starting GRAPH EXPERIMENTS WITH GENDER SIDE INFO ON X MATRIX..."    (tunning all)
#echo "Starting GCE-UII EXPERIMENTS..."

#python main.py --dataset $1 --algo_name mf --uii --epochs $2 --lr 0.01 --batch_size 1024 --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UII_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO"   #done
#python main.py --dataset $1 --algo_name fm --uii --epochs $2 --lr ? --batch_size ? --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UII_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO"  (tunning tune-sideinfo)

#echo "Starting GCE-UIC EXPERIMENTS..."
#python main.py --dataset $1 --algo_name mf --epochs $2 --lr ? --batch_size ? --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_mf_epochs=$2.txt
#echo "DONE MF-GCE-GENDR SIDE-INFO" (tunning tune-sideinfo ....next)
#python main.py --dataset $1 --algo_name fm --epochs $2 --lr ? --batch_size ? --dropout 0 --gce --not_early_stopping --side_information > results/context/outputs_$1/SI_UIC_graph_$1_fm_epochs=$2.txt
#echo "DONE FM-GCE-GENDR SIDE-INFO" (tunning tune-fmGCE-uic)

