#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os

def prepare_dataset(tokenizer, max_length=2048):
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
    
    def format_example(question, cot, response):
        return f"Question: {question}\nReasoning: {cot}\nAnswer: {response}"

    def tokenize_function(examples):
        formatted_texts = [
            format_example(q, c, r)
            for q, c, r in zip(examples["Question"], examples["Complex_CoT"], examples["Response"])
        ]
        return tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['Question', 'Complex_CoT', 'Response']
    )

    return tokenized_dataset


def main():
    model_name = "openai/gpt-oss-20b" 
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  
        device_map="auto",
        trust_remote_code=True
    )


    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./medical_gpt_oss_20b",
        per_device_train_batch_size=1,    
        gradient_accumulation_steps=8,    
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=500,                        
        fp16=False,                              
        bf16=True,                               
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,                      
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./medical_gpt_oss_20b_final")


if __name__ == "__main__":
    main()
