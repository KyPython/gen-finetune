"""
Streamlit web app for the fine-tuned GPT-2 model.

This provides a simple web interface for text generation.
Deploy this to Streamlit Cloud, Heroku, or any Python hosting service.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
MODEL_DIR = "./artifacts"
MAX_LENGTH = 100
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True


@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer (cached)."""
    if not os.path.exists(MODEL_DIR):
        return None, None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def generate_text(model, tokenizer, device, prompt, max_length, temperature, top_p, do_sample):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    st.set_page_config(
        page_title="GPT-2 Fine-Tuned Text Generator",
        page_icon="🤖",
        layout="centered",
    )
    
    st.title("🤖 GPT-2 Fine-Tuned Text Generator")
    st.markdown("Generate text using a fine-tuned GPT-2 model")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        st.warning(
            "⚠️ Model not found. Please run `python src/finetune.py` first to train the model.\n\n"
            "The model will be saved to `./artifacts/` directory."
        )
        st.stop()
    
    st.success(f"✅ Model loaded on device: {device}")
    
    # Input section
    st.header("Input")
    prompt = st.text_area(
        "Enter your prompt:",
        value="The weather today is",
        height=100,
        help="Type the text you want to continue or generate from.",
    )
    
    # Generation parameters
    st.header("Generation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider(
            "Max Length",
            min_value=20,
            max_value=200,
            value=MAX_LENGTH,
            step=10,
            help="Maximum length of generated text",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=TEMPERATURE,
            step=0.1,
            help="Lower = more deterministic, higher = more creative",
        )
    
    with col2:
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=TOP_P,
            step=0.05,
            help="Nucleus sampling parameter",
        )
        do_sample = st.checkbox(
            "Use Sampling",
            value=DO_SAMPLE,
            help="Enable sampling (uncheck for greedy decoding)",
        )
    
    # Generate button
    if st.button("Generate Text", type="primary"):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating text..."):
                try:
                    generated = generate_text(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        max_length,
                        temperature,
                        top_p,
                        do_sample,
                    )
                    
                    st.header("Generated Text")
                    st.text_area(
                        "Output:",
                        value=generated,
                        height=200,
                        key="output",
                        label_visibility="collapsed",
                    )
                except Exception as e:
                    st.error(f"Error during generation: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**About:** This app uses a fine-tuned GPT-2 model. "
        "See the [GitHub repository](https://github.com/YOUR_USERNAME/gen-finetune) for more details."
    )


if __name__ == "__main__":
    main()

