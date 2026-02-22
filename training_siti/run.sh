# script for run the ft 
cd ..
ulimit -c 0 && \
python training_siti/train_siti_agent.py \
    --train-file training_siti/data/train.parquet \
    --val-file training_siti/data/test.parquet \
    --lora \
    --lora-rank 16 \
    --trajectory-level \
    --n-runners 10 \
    --agent-match chat \
    --start-embedding \
    --ci \
    --ci-fast \