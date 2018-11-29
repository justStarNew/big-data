#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=1024
#SBATCH -o outfile
#SBATCH -e errfile
#SBATCH -t 0:11:00
#SBATCH --mail-type=END 
#SBATCH --mail-user=pierre.navaro@univ-rennes2.fr

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate big-data
#python big-data/keras_example.py
python big-data/evaluate_model.py
