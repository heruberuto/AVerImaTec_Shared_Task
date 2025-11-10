export SINGULARITYENV_HOST=0.0.0.0
export SINGULARITYENV_PORT=8479
singularity run \
    --env HOST=0.0.0.0 \
    --env PORT=8479 \
    --bind /mnt/personal/ullriher/open-webui:/app/backend/data \
    /mnt/personal/ullriher/open-webui.sif \
    /app/backend/start.sh