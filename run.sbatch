#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --job-name=python-ex
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda
#SBATCH --output=resnet-ash_p@60
#SBATCH --error=resnet-ash_p@60
###### 1 Load the module
module load nvidia/cudasdk/10.1
module load intel/python/3/2019.4.088
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python ood_eval.py --config config/imagenet_config_ashp.yml --use-gpu
