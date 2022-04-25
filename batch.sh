#!/bin/sh
#nvidia-smi #-q

#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh[00,10-16]

python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate


pip3 install sentence-transformers --user

pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cu111.html --user
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cu111.html --user
pip3 install torch-geometric --user

python3 graph_Generation.py