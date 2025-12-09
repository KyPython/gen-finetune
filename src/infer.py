"""
Simple inference script for the fine-tuned model.

Loads the fine-tuned model and generates text from a prompt.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
MODEL_DIR = "./artifacts"
MAX_LENGTH = 100  # Maximum length of generated text
TEMPERATURE = 0.7  # Lower = more deterministic, higher = more creative
TOP_P = 0.9  # Nucleus sampling parameter
DO_SAMPLE = True  # Set to False for greedy decoding


def load_model_and_tokenizer(model_dir):
    """Load the fine-tuned model and tokenizer."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Please run 'python src/finetune.py' first to train the model."
        )
    
    print(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_length=MAX_LENGTH, 
                  temperature=TEMPERATURE, top_p=TOP_P, do_sample=DO_SAMPLE):
    """Generate text from a prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main inference function."""
    # Load model
    try:
        model, tokenizer, device = load_model_and_tokenizer(MODEL_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get prompt from command line or user input
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"\nPrompt: {prompt}")
    else:
        prompt = input("\nEnter a prompt: ")
    
    if not prompt.strip():
        print("Error: Prompt cannot be empty.")
        sys.exit(1)
    
    # Generate text
    print("\nGenerating text...")
    generated = generate_text(
        model, 
        tokenizer, 
        device, 
        prompt,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
    )
    
    # Print result
    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(generated)
    print("=" * 60)


if __name__ == "__main__":
    main()

