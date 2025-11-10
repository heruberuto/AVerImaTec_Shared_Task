#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=10           # Number of CPU to request for the job
#SBATCH --mem=50GB                   # How much memory does your job require?
#SBATCH --gres=gpu:2            # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --output=_offline.out            # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --constraint=l40
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=xxxx               # The partition you've been assigned
#SBATCH --account=xxxx # The account you've been assigned (normally student)
#SBATCH --qos=xxxx  # What is the QOS assigned to you? Check with myinfo command
#SBATCH --job-name=offline    # Give the job a name

srun python ../prepare_submission/eval_offline.py --llm_name 'gemma' --mllm_name 'gemma' --save_num 12 --root_dir [YOUR_ROOT_DIR] --eval_model "google/gemma-3-27b-it" 
srun python ../prepare_submission/eval_offline.py --llm_name 'gemini-2.0-flash-001' --mllm_name 'gemini-2.0-flash-001' --save_num 12 --root_dir [YOUR_ROOT_DIR] --eval_model "google/gemma-3-27b-it"
srun python ../prepare_submission/eval_offline.py --llm_name 'qwen' --mllm_name 'qwen' --save_num 12 --root_dir [YOUR_ROOT_DIR] --eval_model "google/gemma-3-27b-it" 
srun python ../prepare_submission/eval_offline.py --llm_name 'qwen' --mllm_name 'llava' --save_num 12 --root_dir [YOUR_ROOT_DIR] --eval_model "google/gemma-3-27b-it" 