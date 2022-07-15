#!/bin/bash

# #SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH -o pc-gp-%j-%a.out
#SBATCH -a 0-4
# #SBATCH --exclusive

# module load anaconda/2020b
# conda init
source activate bocf

# python opt_scatter_bocf.py  --results-dir results/opt/scatter-hipass/bocf --trials 1 --trial-i $SLURM_ARRAY_TASK_ID  --objective hipass &
# python opt_scatter_bocf.py  --results-dir results/opt/scatter-orange/bocf --trials 1 --trial-i $SLURM_ARRAY_TASK_ID  --objective orange &
# wait

export OMP_NUM_THREADS=1
python opt_pc_bocf.py  --results-dir results/opt/dos10/bocf --trials 1 --trial-i $SLURM_ARRAY_TASK_ID  --objective dos10 --no-sample-full 
# python opt_pc_bocf.py  --results-dir results/opt/dos10-full/bocf --trials 1 --trial-i $SLURM_ARRAY_TASK_ID --objective dos10 --sample-full
# wait

# python opt_scatter_bocf.py  --results-dir results/opt/scatter-hipass-time/bocf --trials 1  --objective hipass
