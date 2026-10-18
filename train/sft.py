# Supervised fine-tuning on a local jsonl with a 'text' field (loss on the assistant response only).
# Adapted from the s1 training script (github.com/simplescaling/s1, train/sft.py; Apache License 2.0).
# Changes: local jsonl input, Liger fused cross-entropy (avoids materializing full 32k x 152k logits), resume support.
import logging
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import transformers
import trl
from datasets import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-Math-7B")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(default="train.jsonl")
    wandb_project: Optional[str] = field(default="degmatch")


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2()
    logging.info(f"Training config: {({**asdict(config), **asdict(args)})}")

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    dataset = load_dataset("json", data_files={"train": config.train_file_path})
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.pad_token = "<|fim_pad|>"
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
        mlm=False,
    )
    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(model, train_dataset=dataset["train"], eval_dataset=dataset["train"], args=args, data_collator=collator)

    last_ckpt = None
    if os.path.isdir(args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
