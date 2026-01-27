#!/bin/bash
#SBATCH --job-name=AdaIN_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:30:00
#SBATCH --output=output/01_%j.log
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1355863@u.nus.edu

lr=""
epochs=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)
            lr="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

source ./venv/bin/activate

cmd="python main.py \
    --lr \"$lr\" \
    --epochs \"$epochs\" "

eval "$cmd"