
# script for run the ft with sft

model_id="Qwen/Qwen3-4B-Instruct-2507"
# datasets & output dir
train_file="training_siti/data/train_sft_extractor.json"
val_file="training_siti/data/test_sft_extractor.json"
output_dir="checkpoints/SFT/"

lora_rank=16
lr=5e-5
per_device_bs=2
grad_acc_step=4
epochs=6
max_seq_length=1024
eval_per_device_bs=2

# eval & save
eval_steps=10
log_steps=5
save_steps=30
save_tot_limit=3

# optim
optim="adamw_torch_fused"
#  reporting 
run_name="sft_extractor_1"
export WANDB_PROJECT="SitiBTAgentSFT"

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
    --eval-device-batch-size ${eval_per_device_bs} \
    --n-epochs ${epochs} \
    --max-seq-length ${max_seq_length} \
    --eval-steps ${eval_steps} \
    --logging-step ${log_steps} \
    --save-step ${save_steps} \
    --total-ckp-limit ${save_tot_limit} \
    --optim ${optim} \
    --run-name ${run_name} 
    
