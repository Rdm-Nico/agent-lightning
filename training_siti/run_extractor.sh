# script for run the ft 
cd ..
ulimit -c 0 && \
python training_siti/train_siti_agent.py \
    --train-file training_siti/data/train_extractor.parquet \
    --val-file training_siti/data/test_extractor.parquet \
    --lora \
    --lora-rank 16 \
    --lora-adapter-path checkpoints/SitiBTAgent/ft_extractor_3/global_step_96/actor/lora_adapter \
    --n-runners 10 \
    --base-config extractor  