# script for run the ft with sft

model_id="Qwen/Qwen3-4B-Instruct-2507"
# datasets & output dir
train_file="training_siti/data/train_extractor_w_embedding.parquet"
val_file="training_siti/data/test_extractor_w_embedding.parquet"
output_dir="checkpoints/SFT/"

lora_rank=16
lr=2e-4
per_device_bs=2
grad_acc_step=8
epochs=3
max_seq_length=2048

# eval & save
eval_steps=50
log_steps=10
save_steps=50
save_tot_limit=3

# optim
optim="adamw_torch_fused"
#  reporting 
run_name="sft_agent_1"
project_name="SitiBTAgentSFT"

cd ..
ulimit -c 0 && \
python training_siti/sft_train.py \
    --model "${model_id}" \
    --train-file "${train_file}"  \
    --val-file  "${val_file}"\
    --output-dir "${output_dir}"\
    --lora \
    --lora-rank ${lora_rank} \
    --lr ${lr} \
    --device-batch-size ${per_device_bs} \
    --gradient-accumulation-steps ${grad_acc_step} \
    --n-epochs ${epochs} \
    --max-seq-length ${max_seq_length} \
    --eval-steps ${eval_steps} \
    --logging-step ${log_steps} \
    --save-step ${save_steps} \
    --total-ckp-limit ${save_tot_limit} \
    --optim ${optim} \
    --run-name ${run_name} \
    --project-name ${project_name} \






    