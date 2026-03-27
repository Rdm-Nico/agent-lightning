from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from utils.logger import Logger
import os
logger = Logger(save=False, consoleLevel="INFO").getLogger()



def main():

    # get config
    parser = argparse.ArgumentParser(description="Train a tool/calling agent with SFT.")
    parser.add_argument("--train-file", type=str, default="data/train_sft.json", help="Path to train json file")
    parser.add_argument("--val-file", type=str, default="data/test_sft.json", help="Path to val json file")
    parser.add_argument("--output-dir", type=str, default="../checkpoints/SFT/", help="Path  of the output dir")
    parser.add_argument("--project-name", type=str, default=None, help="Project name in which to save the checkpoints and wandb")
    parser.add_argument("--run-name", type=str, default=None, help="Run name of wandb")
    parser.add_argument("--model", type=str, default=None, help="HF model id or path (optional)")
    parser.add_argument("--n-epochs", type=float, default=3.0, help="Number of train epochs")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1.0, help="Number of update steps to accumulate gradients before performing a backward pass")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping applied after backward pass before optimizer step for preventing gradient explosion")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0, help="Label smoothing to prevent overconfidence")
    parser.add_argument("--logging-step", type=int, default=1, help="Number of update steps between two logs if logging_strategy='steps' ")
    parser.add_argument("--logging-level", type=str, default="passive", help="Logging level")
    parser.add_argument("--eval-steps", type=int, default=15, help="Number of update steps between two evaluations")
    parser.add_argument("--eval-device-batch-size", type=int, default=16, help="Per device eval batch size")
    parser.add_argument("--save-step", type=int, default=20, help="Number of updates steps before two checkpoint saves")
    parser.add_argument("--total-ckp-limit", type=int, default=4, help="Maximum number of checkpoints to keep. Deletes older checkpoints. If load_best_model_at_end=True, the best checkpoint is always retained plus the most recent ones")
    parser.add_argument("--load-best-ckp", type=bool, default=True, help="Load the best checkpoints at the end of the training. When `True`, `save_strategy` must match `eval_strategy`, and if using `steps`, `save_steps` must be a multiple of `eval_steps` ")
    parser.add_argument("--resum-from-checkpoint", type=str,help="Path to a folder with a valid checkpoint ")
    parser.add_argument("--max-seq-length", type=int, default=20, help="max sequence length")
    parser.add_argument("--optim", type=str, help="optimizer")
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
    # train sft agent
    dataset_train = load_dataset('json', data_files=args.train_file)
    dataset_val = load_dataset('json', data_files=args.val_file)
    logger.info(f"len of dataset_train: {dataset_train.shape}")
    logger.info(f"len of dataset_val: {dataset_val.shape}")


    tokeinzer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ) 

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
            ]
        )
    
    # add sft training
    if args.lr:
        if args.project_name == None:
            logger.error(f"errore: non é specificato il nome del progetto")
            return 
        
        output_path = os.path.join(args.output_dir, args.project_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)


        train_args = SFTConfig(
            assistant_only_loss=True,
            loss_type="dft",
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
            torch_compile=True,
            auto_find_batch_size=True,
            log_level=args.logging_level,
            report_to=["wandb"],
            run_name=args.run_name,
            project=args.project_name,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            eval_on_start=True,
            per_device_eval_batch_size=args.eval_device_batch_size,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.total_ckp_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
            resume_from_checkpoint=args.resum_from_checkpoint,
            packing=False,
            max_seq_length=args.max_seq_length
        )


if '__main__' == __name__:
    main()