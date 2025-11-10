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
#SBATCH --gres=gpu:2             # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --output=_gemini.out            # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --constraint=a100
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=xxxx               # The partition you've been assigned
#SBATCH --account=xxxx # The account you've been assigned (normally student)
#SBATCH --qos=xxxx  # What is the QOS assigned to you? Check with myinfo command
#SBATCH --job-name=baseline    # Give the job a name

srun python ../src/mm_checker.py --LLM_NAME 'gemini-2.0-flash-001' --MLLM_NAME 'gemini-2.0-flash-001' --SAVE_NUM 1 --DATA_STORE True --ROOT_PATH  [YOUR_ROOT_DIR] --DATASTORE_PATH [YOUR_DATASTORE_PATH]
srun python ../src/mm_checker.py --LLM_NAME 'gemini-2.0-flash-001' --MLLM_NAME 'gemini-2.0-flash-001' --SAVE_NUM 2 --DATA_STORE True --ROOT_PATH  [YOUR_ROOT_DIR] --DATASTORE_PATH [YOUR_DATASTORE_PATH] --PARA_QG True

srun python ../src/mm_checker.py --LLM_NAME 'gemma' --MLLM_NAME 'gemma' --SAVE_NUM 1 --DATA_STORE True --ROOT_PATH  [YOUR_ROOT_DIR] --DATASTORE_PATH  [YOUR_DATASTORE_PATH]
srun python ../src/mm_checker.py --LLM_NAME 'gemma' --MLLM_NAME 'gemma' --SAVE_NUM 2 --DATA_STORE True --ROOT_PATH  [YOUR_ROOT_DIR] --DATASTORE_PATH  [YOUR_DATASTORE_PATH] --PARA_QG True