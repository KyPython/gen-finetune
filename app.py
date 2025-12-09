"""
Streamlit web app for the fine-tuned GPT-2 model.

This provides a simple web interface for text generation.
Deploy this to Streamlit Cloud, Heroku, or any Python hosting service.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import subprocess

# Configuration - Use absolute path based on app location
MODEL_DIR = str(Path(__file__).parent / "artifacts")
MAX_LENGTH = 100
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True


def train_model_if_needed():
    """Train the model if it doesn't exist."""
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(MODEL_DIR) and os.path.exists(config_path):
        return True  # Model already exists
    
    try:
        # Run the training script as a subprocess
        script_path = Path(__file__).parent / "src" / "finetune.py"
        if not script_path.exists():
            st.error(f"Training script not found at: {script_path}")
            return False
            
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=Path(__file__).parent  # Run from app directory
        )
        
        if result.returncode == 0:
            # Verify model was created
            if os.path.exists(MODEL_DIR) and os.path.exists(config_path):
                return True
            else:
                st.error(f"Training completed but model not found at {MODEL_DIR}")
                return False
        else:
            st.error(f"Training failed: {result.stderr}")
            if result.stdout:
                st.text("Training output:")
                st.text(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        st.error("Training timed out. Please train the model locally first.")
        return False
    except Exception as e:
        st.error(f"Error training model: {e}")
        return False


@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer (cached)."""
    # Lazy import to avoid startup delays
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # Check if model exists
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(MODEL_DIR):
        return None, None, None
    if not os.path.exists(config_path):
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
    import torch
    
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
    
    # Initialize session state for model loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'device' not in st.session_state:
        st.session_state.device = None
    
    # Quick check if model exists (no loading - just check file system)
    config_path = os.path.join(MODEL_DIR, "config.json")
    model_exists = os.path.exists(MODEL_DIR) and os.path.exists(config_path)
    
    # Show status message
    if st.session_state.model_loaded:
        st.success(f"✅ Model loaded on device: {st.session_state.device}")
    elif model_exists:
        st.info("ℹ️ Model found. Click 'Generate Text' to load and use it.")
    else:
        st.info("📝 Model not found. It will be trained automatically when you first generate text (may take a few minutes).")
    
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
            # Load or train model if not already loaded (lazy loading)
            if not st.session_state.model_loaded:
                config_path = os.path.join(MODEL_DIR, "config.json")
                model_exists = os.path.exists(MODEL_DIR) and os.path.exists(config_path)
                
                if not model_exists:
                    # Train the model
                    with st.spinner("Training model... This is a one-time setup (may take a few minutes)."):
                        success = train_model_if_needed()
                    
                    if not success:
                        st.error(
                            "❌ Failed to train model. Please ensure all dependencies are installed.\n\n"
                            "You can also train locally by running: `python src/finetune.py`"
                        )
                        st.stop()
                    
                    # Clear cache to reload after training
                    load_model_and_tokenizer.clear()
                
                # Load the model (either newly trained or existing)
                with st.spinner("Loading model..."):
                    model, tokenizer, device = load_model_and_tokenizer()
                    if model is None:
                        st.error("Failed to load model. Please refresh the page.")
                        st.stop()
                    
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.device = device
                    st.session_state.model_loaded = True
            
            # Generate text (model is now guaranteed to be loaded)
            with st.spinner("Generating text..."):
                try:
                    generated = generate_text(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.device,
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
        "See the [GitHub repository](https://github.com/KyPython/gen-finetune) for more details."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        raise

