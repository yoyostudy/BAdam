#################
### Code adapted from https://github.com/RobinSmits/BAdam-Qwen

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from trl import SFTTrainer
from src.badam.block_optim import BlockOptimizer, BlockOptimizerRatio

def main():
    model_name = 'Qwen/Qwen1.5-1.8B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    MAX_LEN = 1024
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map = "auto",
                                             torch_dtype = torch.bfloat16)
    test_data = load_dataset('BramVanroy/ultrachat_200k_dutch', split = 'test_sft')
    test_data = test_data.select(range(3072)).remove_columns(["prompt", "prompt_id"]) # Only part of the test data to limit the time spent on validation.
    train_data = load_dataset('BramVanroy/ultrachat_200k_dutch', split = 'train_sft').remove_columns(["prompt", "prompt_id"])
    optimizer = BlockOptimizer(base_optimizer = torch.optim.AdamW(model.parameters(), lr = 5.0e-5, weight_decay = 0.001),
                           named_parameters_list = list(model.named_parameters()),
                           block_prefix_list = None,
                           switch_block_every = 16,
                           switch_mode = "descending",
                           verbose = 0)
    # Set Steps
    eval_steps = 256
    save_steps = 1024
    logging_steps = 128

    # Set TrainingArguments
    training_args = TrainingArguments(
                                    output_dir = './output',
                                    num_train_epochs = 1,
                                    max_steps = 8192,
                                    evaluation_strategy = "steps",
                                    logging_steps = logging_steps,
                                    save_strategy = 'steps',
                                    eval_steps = eval_steps,
                                    save_steps = save_steps,
                                    save_total_limit = 1,
                                    per_device_train_batch_size = 2,
                                    per_device_eval_batch_size = 2,
                                    gradient_accumulation_steps = 8,
                                    gradient_checkpointing = True,
                                    gradient_checkpointing_kwargs = {'use_reentrant': False},
                                    bf16 = True)

    # Config SFTTrainer
    trainer = SFTTrainer(model,
                        train_dataset = train_data,
                        eval_dataset = test_data,
                        tokenizer = tokenizer,
                        packing = True,
                        eval_packing = False,
                        max_seq_length = MAX_LEN,
                        optimizers = (optimizer, None),
                        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
                        args = training_args,
                        )

    # Perform Training
    trainer.train()

if __name__ == "__main__":
    main()
