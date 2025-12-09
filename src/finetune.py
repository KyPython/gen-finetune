"""
Minimal fine-tuning script for a small causal language model.

This script fine-tunes a small GPT-2 model on a tiny toy dataset.
For production use, replace the toy dataset with a larger dataset
and consider using multiple GPUs with Accelerate or Ray.
"""

import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import torch

# Configuration
MODEL_NAME = "gpt2"  # Change to "gpt2-medium", "gpt2-large", etc. for larger models
OUTPUT_DIR = "./artifacts"
MAX_STEPS = 50  # Small number for quick CPU training; increase for better results
BATCH_SIZE = 2  # Small batch size for CPU; increase if using GPU
LEARNING_RATE = 5e-5


def create_toy_dataset():
    """
    Create a tiny toy dataset of prompt-response pairs.
    
    In production, replace this with:
    - Loading from Hugging Face datasets: datasets.load_dataset("dataset_name")
    - Loading from a CSV/JSON file
    - Loading from a custom data loader
    """
    # Tiny dataset of simple Q&A style prompts
    texts = [
        "The weather today is sunny and warm.",
        "Python is a popular programming language.",
        "Machine learning models require data to train.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Transformers are powerful neural network architectures.",
        "Natural language processing enables computers to understand text.",
        "Deep learning uses multiple layers to learn representations.",
        "Gradient descent optimizes model parameters during training.",
        "Tokenization converts text into numerical representations.",
        "Attention mechanisms help models focus on relevant parts of input.",
    ]
    
    # Convert to dataset format
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset for causal language modeling."""
    def tokenize_function(examples):
        # Tokenize text and set labels (for causal LM, labels are the same as input_ids)
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,  # Adjust based on your data
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


def main():
    """Main fine-tuning function."""
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Set pad token if not already set (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Creating toy dataset...")
    dataset = create_toy_dataset()
    
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Split into train (and optionally eval) - for this minimal example, use all for training
    train_dataset = tokenized_dataset
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=MAX_STEPS,  # Quick training for CPU
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=25,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        # For GPU training, uncomment and configure:
        # fp16=True,  # Enable mixed precision training
        # dataloader_num_workers=4,  # Parallel data loading
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    print(f"Starting training for {MAX_STEPS} steps...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Train
    train_result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Save model and tokenizer
    print(f"Saving model and tokenizer to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

