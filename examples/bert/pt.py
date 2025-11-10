"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/bert/pt.py

- 8 GPUs (DDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml \
        examples/bert/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 8 GPUs (DDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "ddp" \
        --script_path "examples/bert/pt.py"
"""



import os
import functools
from dataclasses import dataclass, field

import transformers
import accelerate
from peft import LoraConfig, get_peft_model

import dllm


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "jhu-clsp/mmBERT-base"
    use_lora: bool = field(default=True, metadata={"help": "Apply LoRA adapters"})


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "Q-bert/wiki-tr-filtered"
    text_field: str = "text"
    max_length: int = 8192
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/mmbert-turkish-wiki"
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_steps: float = 0.1
    save_steps: float = 0.1


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)

    # ----- LoRA Uygulama ----------------------------------------------------------
    if getattr(model_args, "use_lora", False):
        lora_config = LoraConfig(
                        r=128,
                        lora_alpha=128,
                        target_modules = [
                            "attn.Wqkv",
                            "attn.Wo",
                            "mlp.Wi",
                            "mlp.Wo",
                        ],
                        lora_dropout=0.1,
                        bias="none",
                    )
        model = get_peft_model(model, lora_config)
        dllm.utils.print_main("LoRA adapters applied.")
        model.print_trainable_parameters()

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
        )
        dataset = dataset.map(
            functools.partial(
                dllm.utils.tokenize_and_group,
                tokenizer=tokenizer,
                text_field=data_args.text_field,
                seq_length=data_args.max_length,
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail,
            ),
            batched=True,
            num_proc=None if data_args.streaming else data_args.num_proc,
            remove_columns=dataset["train"].column_names,
        )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train()

    # ----- Save ------------------------------------------------------------------
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    dllm.utils.print_main(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    train()
