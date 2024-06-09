#################
### Code adapted from https://github.com/RobinSmits/BAdam-Qwen

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer ## 
)
from trl import SFTTrainer
from src.badam.block_optim import BlockOptimizer, BlockOptimizerRatio


### FOR DEBUG
from transformers.modeling_utils import unwrap_model
from functools import wraps
import pdb


class cSFTTrainer(SFTTrainer):
    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        #pdb.set_trace()
        for param in self.model.parameters():
            #print(f"Parameter requires_grad: {param.requires_grad}, Parameter data: {param.data}")
            param.requires_grad = True
            #print(f"Parameter requires_grad: {param.requires_grad}, Parameter data: {param.data}")

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            # if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            #     embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            # else:
            #     embeddings = unwrapped_model.get_input_embeddings()
            embeddings = unwrap_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

def main():
    model_name = 'Qwen/Qwen1.5-1.8B-Chat'
    #model_name = "THUDM/chatglm2-6b" # NOT WORKING
    #model_name = "meta-llama/Meta-Llama-3-8B" # TODO: To be test
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True)
    MAX_LEN = 100
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map = "auto",
                                             torch_dtype = torch.bfloat16)
    # model = AutoModel.from_pretrained(
    #     model_name,
    #     trust_remote_code = True,
    #     device_map = "cuda",
    #     torch_dtype = torch.bfloat16
    # )
    

    print(model)

    datasets = load_dataset("FinGPT/fingpt-sentiment-train")
    train_data = datasets['train']
    test_data = datasets['train']
    optimizer = BlockOptimizer(
        base_optimizer = torch.optim.AdamW(model.parameters(), lr = 5.0e-5, weight_decay = 0.001),
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
                                    output_dir = 'output',
                                    num_train_epochs = 1,
                                    max_steps = 8192,
                                    evaluation_strategy = "steps",
                                    logging_steps = logging_steps,
                                    save_strategy = 'steps',
                                    eval_steps = eval_steps,
                                    save_steps = save_steps,
                                    save_total_limit = 1,
                                    per_device_train_batch_size = 1,
                                    per_device_eval_batch_size = 1,
                                    gradient_accumulation_steps = 1,
                                    gradient_checkpointing = True,
                                    gradient_checkpointing_kwargs = {'use_reentrant': False},
                                    bf16 = True)

    # Config SFTTrainer
    trainer = cSFTTrainer(model,
                        train_dataset = train_data,
                        eval_dataset = test_data,
                        tokenizer = tokenizer,
                        packing = True,
                        eval_packing = False,
                        max_seq_length = MAX_LEN,
                        optimizers = (optimizer, None),
                        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
                        args = training_args,
                        dataset_text_field='instruction'
                        )

    # Perform Training
    trainer.train()

if __name__ == "__main__":
    main()
