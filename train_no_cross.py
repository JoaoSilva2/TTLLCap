from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers import set_seed, TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, default_data_collator, TrainingArguments

import pandas as pd
import numpy as np
import torch
import random
import os
import argparse
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformers import default_data_collator

from transformers import set_seed

from src.utils import *

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

#----------------- Seeds -----------------#
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
set_seed(0)
#-----------------------------------------#


class EpochSaveCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.bs = kwargs['batch_size']

        self.checkpoint = 8856
        if kwargs['batch_size'] != 64:
            self.checkpoint = 8855
        
        self.model_path = kwargs['model_path']

    def on_log(self, args, state, control, logs=None, **kwargs):
        kwargs["model"].save_pretrained(f"{self.model_path}/checkpoint-{self.checkpoint}")

        if self.bs == 64:
            self.checkpoint += 8856
        else:
            self.checkpoint += 8855

#----------------------- DATASET SETUP ---------------------#
class MyTrainDataset(Dataset):
    def __init__(self, df, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.max_target_length = max_caption_length

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                    + max_caption_length * k # retrieved captions
                                    + len(tokenizer.encode(self.template)) # template
                                    + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                    )

            assert k is not None
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)

        # load precomputed features
        encoding = {"input_ids": torch.tensor(decoder_input_ids), "labels": torch.tensor(labels)}

        return encoding
#---------------------------------------------------------#


#----------------------- SETUP MODEL ---------------------#
def main(args):
    lora_config = LoraConfig(
        use_rslora=True,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.decoder_name)
    peft_model = get_peft_model(base_model, lora_config)  

    peft_model.print_trainable_parameters()

    trainable_params = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()

    print(f"Actual trainable parameters: {trainable_params}")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    #---------------------------------------------------------#

    #----------------------- SETUP DATA ---------------------_#

    data = load_data_for_training(args.annotations_path, args.captions_path, args.categories_path)
    train_df = pd.DataFrame(data['train'])

    train_dataset = MyTrainDataset(
                                df=train_df,
                                tokenizer=tokenizer,
                                rag=True,
                                template_path=args.template_path,
                                k=4,
                                max_caption_length=25)

    training_args = TrainingArguments(
        num_train_epochs=args.n_epochs, 
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate = args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs, 
        logging_strategy="epoch", 
        output_dir=args.experiments_path, 
        overwrite_output_dir=True, 
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset,
    )

    callback = EpochSaveCallback(batch_size=args.batch_size, model_path=args.experiments_path)

    trainer.add_callback(callback)
    trainer.train()
    
    for i in range(args.n_epochs):
        if args.batch_size == 64:
            checkpoint = 8856 * (i+1)
        else:
            checkpoint = 8855 * (i+1)

        base_model = AutoModelForCausalLM.from_pretrained(args.decoder_name)
        peft_model = PeftModel.from_pretrained(base_model, f"{args.experiments_path}/checkpoint-{checkpoint}")
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(f"{args.experiments_path}/checkpoint-{checkpoint}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--experiments_path", type=str, default="exp_2024/", help="Directory where trained models will be saved")

    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps/ViT-L-14/retrieved_caps.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")

    # Socratic model extension
    parser.add_argument("--socratic_models", action="store_true", default=False, help="Whether to use socratic models prompts")
    parser.add_argument("--pk", type=int, default=0, help="Number of place categories to be added to the prompt")
    parser.add_argument("--ok", type=int, default=0, help="NUmber of object categories to be added to the prompt")
    parser.add_argument("--categories_path", type=str, default="data/retrieved_cats/ViT-L-14/retrieved_categories.json", help="JSON file with retrieved captions")

    parser.add_argument("--rank", type=int, default=4, help="Value of rank used in LoRA (higher - more parameters)")
    parser.add_argument("--alpha", type=int, default=8, help="Value of alpha used in LoRA (higher - less impact of fine-tuning)")

    args = parser.parse_args()

    main(args)