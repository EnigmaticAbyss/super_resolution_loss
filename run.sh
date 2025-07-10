#!/bin/bash -l

#SBATCH --job-name=SuperRes

#SBATCH --partition=a100
#SBATCH --clusters=tinygpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --export=NONE
#SBATCH --mail-user=arashmousavi193@gmail.com
#SBATCH --mail-type=ALL

                             # first non-empty non-comment line ends SBATCH options
. ~/.bashrc # if you dont work with conda, comment this 
source activate SuperRes #and this
         # Setup job environment (load modules, stage data, ...)

python3 -u ./main.py       # Execute parallel application