#!/usr/bin/env python3
"""
RCH-StackBot-3.8B Training Script
Tränar Phi-3 Mini för svenska fullstack-utveckling
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse

def setup_model_and_tokenizer(model_name="microsoft/Phi-3-mini-4k-instruct"):
    """Ladda och konfigurera modell och tokenizer"""
    print(f"Laddar modell: {model_name}")
    
    # Ladda tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ladda modell med kvantisering för att spara minne
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora_config():
    """Konfigurera LoRA för effektiv fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    return lora_config

def prepare_dataset(tokenizer, data_path="data/training_data.txt"):
    """Förbered träningsdata"""
    print(f"Laddar träningsdata från: {data_path}")
    
    # Ladda data
    if data_path.endswith('.txt'):
        dataset = load_dataset('text', data_files=data_path)
    else:
        dataset = load_dataset(data_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Träna RCH-StackBot-3.8B")
    parser.add_argument("--data_path", default="data/training_data.txt", help="Sökväg till träningsdata")
    parser.add_argument("--output_dir", default="./models/RCH-StackBot-3.8B", help="Utdata-mapp")
    parser.add_argument("--epochs", type=int, default=3, help="Antal tränings-epoker")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch-storlek")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Inlärningshastighet")
    
    args = parser.parse_args()
    
    # Skapa utdata-mapp
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ladda modell och tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Konfigurera LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"Modell har {model.num_parameters()} parametrar")
    print(f"Träningsbara parametrar: {model.num_parameters(only_trainable=True)}")
    
    # Förbered dataset
    dataset = prepare_dataset(tokenizer, args.data_path)
    
    # Träningsargument
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="epoch",
        save_steps=100,
        evaluation_strategy="no",
        load_best_model_at_end=False,
        report_to="tensorboard",
        run_name="RCH-StackBot-3.8B-training"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    
    # Starta träning
    print("Startar träning...")
    trainer.train()
    
    # Spara modell
    print(f"Sparar modell till: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Spara träningskonfiguration
    config = {
        "model_name": "RCH-StackBot-3.8B",
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "training_args": vars(args),
        "lora_config": lora_config.__dict__
    }
    
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Träning klar!")

if __name__ == "__main__":
    main()