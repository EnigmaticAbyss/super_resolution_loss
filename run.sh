#!/bin/bash -l                     # Interpreter directive; -l is necessary to initialize modules correctly!
#

#SBATCH --partition=rtx2080ti 
#SBATCH --gres=gpu:1      #All #SBATCH lines have to follow uninterrupted
#SBATCH --time=6:00:00            
#SBATCH --job-name=jobofmine
#SBATCH --mail-user=arashmousavi193@gmail.com            # do not export environment from submitting shell
#SBATCH --mail-type=ALL
#SBATCH --export=NONE                                       # first non-empty non-comment line ends SBATCH options
unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
eval "$(conda shell.bash hook)" 
conda activate SuperRes
         # Setup job environment (load modules, stage data, ...)

python3 -u ./main.py       # Execute parallel application