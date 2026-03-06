# script for run the ft 
cd ..
ulimit -c 0 && \
python training_siti/train_siti_agent.py \
    --train-file training_siti/data/train_agent_w_embedding.parquet \
    --val-file training_siti/data/test_agent_w_embedding.parquet \
    --lora \
    --lora-rank 16 \
    --trajectory-level \
    --n-runners 10 \
    --base-config agent \
    --agent-match chat \
    --start-embedding \