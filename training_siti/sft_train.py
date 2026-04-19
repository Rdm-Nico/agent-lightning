from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from utils.logger import Logger
import os
import torch
logger = Logger(save=False, consoleLevel="INFO").getLogger()

SYS_PROMPT="Sei un assistente che estrae informazioni. Quando l'utente ti fornisce informazioni riguardanti le ore di lavoro fatte oppure le inefficienze trovate, DEVI chiamare la funzione extractor_expert per estrapolare le informazioni.\nDevi mandare alla funzione extractor_expert un riassunto di quello che l'utente ti ha detto riguardanti le ore di lavoro e le inefficienze trovate.\nDevi essere informale e non giudicare l'utente per le informazioni che lui ti da. Quando prendi in ingresso la risposta del extractor_expert non aggiungere informazioni sbagliate o pareri, devi ritornare quello che extractor_expert ritorna a te.\nDopo aver ricevuto la risposta dal extractor_expert chiedi conferma all'utente per salvare i dati. Se e solo se l'utente conferma le informazioni che sono state estratte te DEVI chiamare\nla funzione push_data per salvare le informazioni nel database, SOLO dopo l'utente ti a confermato che vanno bene, non prima  \nESEMPIO:\n[USER]: \"Ieri ho fatto 7 ore di lavoro\"\n[ASSISTENTE]: <chiama function extractor_expert con parametro: 'lavorato 7 ore '> \n[ASSISTENTE]: <extractor_expert ritorna -> {\n  \"ORE_ORDINARIE\": 7.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"INEFFICIENCY\": false,\n  \"NOTE\": null,\n  \"COMMESSA\": null,\n  \"risposta_singola\":\"\"\n} >\n[ASSISTENTE]: hai fatto 7 ore di lavoro senza inefficienze identificate, è corretto ?\n[USER]: si va bene \n[ASSISTENTE]: <chiama la function push_data>\n[ASSISTENTE]: <extractor_expert ritorna -> La commessa  è stata salvata >\n[ASSISTENTE]: la commessa è stata salvata sul database"




def main():

    # get config
    parser = argparse.ArgumentParser(description="Train a tool/calling agent with SFT.")
    parser.add_argument("--train-file", type=str, default="data/train_sft.json", help="Path to train json file")
    parser.add_argument("--val-file", type=str, default="data/test_sft.json", help="Path to val json file")
    parser.add_argument("--output-dir", type=str, default="../checkpoints/SFT/", help="Path  of the output dir")
    parser.add_argument("--run-name", type=str, default=None, help="Run name of wandb")
    parser.add_argument("--model", type=str, default=None, help="HF model id or path (optional)")
    parser.add_argument("--n-epochs", type=float, default=3.0, help="Number of train epochs")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of update steps to accumulate gradients before performing a backward pass")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping applied after backward pass before optimizer step for preventing gradient explosion")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0, help="Label smoothing to prevent overconfidence")
    parser.add_argument("--logging-step", type=int, default=1, help="Number of update steps between two logs if logging_strategy='steps' ")
    parser.add_argument("--logging-level", type=str, default="passive", help="Logging level")
    parser.add_argument("--eval-steps", type=int, default=15, help="Number of update steps between two evaluations")
    parser.add_argument("--eval-device-batch-size", type=int, default=16, help="Per device eval batch size")
    parser.add_argument("--save-steps", type=int, default=30, help="Number of updates steps before two checkpoint saves")
    parser.add_argument("--total-ckp-limit", type=int, default=4, help="Maximum number of checkpoints to keep. Deletes older checkpoints. If load_best_model_at_end=True, the best checkpoint is always retained plus the most recent ones")
    parser.add_argument("--no-load-best-ckp", action="store_false", dest="load_best_ckp", help="Load the best checkpoints at the end of the training. When `True`, `save_strategy` must match `eval_strategy`, and if using `steps`, `save_steps` must be a multiple of `eval_steps` ")
    parser.set_defaults(load_best_ckp=True)
    parser.add_argument("--resum-from-checkpoint", type=str,help="Path to a folder with a valid checkpoint ")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--optim", type=str, help="optimizer")
    parser.add_argument("--torch-compile", action="store_true", default=False)
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank to use when --lora is enabled (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate (default: 1e-4)",
    )


    args = parser.parse_args()
    logger.info(f"training arguments:\n {args}\n")
    # train sft agent
    dataset_train = load_dataset('json', data_files=args.train_file)["train"]
    dataset_val = load_dataset('json', data_files=args.val_file)["train"]
    logger.info(f"len of dataset_train: {dataset_train.shape}")
    logger.info(f"len of dataset_val: {dataset_val.shape}")


    #tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    #tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Instruct-2507")
    
    
    def format_f(sample):
        """Format function"""
        messages = sample["messages"]
        # add system prompt
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": SYS_PROMPT}] + messages

        return {"messages": messages}
    
    #dataset_train = dataset_train.map(format_f)

    #dataset_val = dataset_val.map(format_f)

    # debug
    """ print(tokenizer.apply_chat_template(
    dataset_val[0]["messages"],
    tools=dataset_val[0].get("tools"),
    tokenize=False,
    add_generation_prompt=False,
    )) """

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.enable_input_require_grads()
    

    if args.lora:
        # add lora config
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[               
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM
        )
    
    

    # add sft training
    
    output_path = os.path.join(args.output_dir, args.run_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_args = SFTConfig(
        assistant_only_loss=False,
        gradient_checkpointing=True,
        bf16=True,
        output_dir=output_path,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing_factor,
        torch_compile=args.torch_compile,
        auto_find_batch_size=False,
        log_level=args.logging_level,
        run_name=args.run_name,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=False,
        per_device_eval_batch_size=args.eval_device_batch_size,
        save_strategy="steps",
        report_to=["wandb"],
        save_steps=args.save_steps,
        save_total_limit=args.total_ckp_limit,
        metric_for_best_model="eval_loss",
        resume_from_checkpoint=args.resum_from_checkpoint,
        packing=False,
        optim=args.optim if args.optim else "adamw_torch_fused",
        load_best_model_at_end=args.load_best_ckp,
        max_length=2048,
        logging_steps=args.logging_step,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        peft_config=lora_config
    )
    
    trainer.train()
        


if '__main__' == __name__:
    main()