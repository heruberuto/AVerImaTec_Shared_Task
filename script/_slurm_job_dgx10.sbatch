#!/bin/bash
#SBATCH --partition interactive
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 128G
#SBATCH --gres gpu:1
#SBATCH --time 72:00:00
#SBATCH --job-name averitec
#SBATCH --output logs/jupyter.%j.out
unset LMOD_ROOT; unset MODULESHOME; unset LMOD_PKG; unset LMOD_CMD; unset LMOD_DIR; unset FPATH; unset __LMOD_REF_COUNT_MODULEPATH; unset __LMOD_REF_COUNT__LMFILES_; unset _LMFILES_; unset _ModuleTable001_; unset _ModuleTable002_
source /etc/profile.d/lmod.sh

# Replace with your own virtual environment
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
ml OpenSSL/3

source /mnt/personal/ullriher/venvs/aug25/bin/activate

# export SSL_CERT_FILE=~/ollama.crt
PORT=$(shuf -i 20000-40000 -n 1)
while lsof -i TCP:$PORT &>/dev/null; do
    PORT=$(shuf -i 20000-40000 -n 1)
done
echo "Ollama server will run on port: $PORT"
HOSTNAME=$(hostname -s)
# Export the port so your Python app can read it
export OLLAMA_PORT=$PORT
export OLLAMA_HOST="http://$HOSTNAME:$PORT"

# Save to a file so your Python process can also load it if needed
echo $PORT > logs/ollama.${SLURM_JOB_ID}.log

nohup ollama serve > logs/ollama.${SLURM_JOB_ID}.log 2>&1 &

# load your .env    
export $(grep -v '^#' .env | xargs)

export PYTHONPATH=src:$PYTHONPATH
jupyter notebook --no-browser --port=$(shuf -i8000-9999 -n1) --ip=$(hostname -s)
