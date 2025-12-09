# Generative Text Model Fine-Tuning

A minimal project for fine-tuning a small causal language model (GPT-2) on a custom dataset and running inference.

## Setup

### 1. Install Dependencies

```bash
pip install transformers datasets torch
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Usage

### Fine-Tuning

Run the fine-tuning script to train the model on a toy dataset:

```bash
python src/finetune.py
```

This will:
- Load a small GPT-2 model
- Create a tiny toy dataset (10 examples)
- Fine-tune for 50 steps (quick CPU training)
- Save the model and tokenizer to `./artifacts`

**Note**: The training is configured for quick CPU execution. For better results:
- Increase `MAX_STEPS` in `src/finetune.py`
- Use a larger dataset (see "Customizing the Dataset" below)
- Enable GPU training with `fp16=True` in training arguments

### Inference

Run the inference script to generate text:

```bash
# Using command line argument
python src/infer.py "The weather today is"

# Or run interactively (will prompt for input)
python src/infer.py
```

The script will:
- Load the fine-tuned model from `./artifacts`
- Generate text from your prompt
- Print the generated output

## Customization

### Changing the Dataset

In `src/finetune.py`, replace the `create_toy_dataset()` function:

```python
def create_toy_dataset():
    # Option 1: Load from Hugging Face datasets
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
    return dataset
    
    # Option 2: Load from a file
    # dataset = Dataset.from_json("data.json")
    # return dataset
    
    # Option 3: Use your custom data
    # texts = ["your", "text", "data", "here"]
    # return Dataset.from_dict({"text": texts})
```

### Changing the Model

In `src/finetune.py`, change the `MODEL_NAME`:

```python
MODEL_NAME = "gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
```

For other models, you can use:
- `"distilgpt2"` - Smaller, faster version
- `"EleutherAI/gpt-neo-125M"` - Alternative small model
- Any other causal LM from Hugging Face

### GPU Training

To use GPU, ensure CUDA is available and modify training arguments in `src/finetune.py`:

```python
training_args = TrainingArguments(
    # ... existing args ...
    fp16=True,  # Enable mixed precision
    dataloader_num_workers=4,  # Parallel data loading
)
```

### Multi-GPU / Distributed Training

For multiple GPUs, use Hugging Face Accelerate:

```bash
pip install accelerate
accelerate config  # Configure your setup
accelerate launch src/finetune.py
```

Or use Ray for distributed training (see Hugging Face Transformers documentation).

## Web Deployment

### Streamlit Web App

A simple web interface is included for public deployment.

#### Local Testing

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run app.py
```

#### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub (already done if you cloned this repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `KyPython/gen-finetune`
6. Set Main file path: `app.py`
7. Click "Deploy"

**Note**: For Streamlit Cloud, you'll need to:
- Train the model locally and commit the `artifacts/` folder, OR
- Add a step in the app to download/train the model on first run
- Update the GitHub URL in `app.py` (line with `YOUR_USERNAME`)

#### Alternative Deployment Options

- **Heroku**: Use `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Railway**: Similar to Heroku, supports Python apps
- **Hugging Face Spaces**: Upload as a Streamlit Space
- **Docker**: Containerize the app for any cloud provider

## Project Structure

```
.
├── src/
│   ├── finetune.py    # Fine-tuning script
│   └── infer.py       # Inference script
├── app.py             # Streamlit web app
├── artifacts/         # Saved model and tokenizer (created after training)
├── requirements.txt
├── streamlit_requirements.txt
├── README.md
└── .streamlit/
    └── config.toml    # Streamlit configuration
```

## Notes

- The default configuration is optimized for quick CPU training
- Training loss is logged every 10 steps
- The model is saved to `./artifacts` after training
- For production use, increase training steps and use a larger, more diverse dataset
- The web app requires the model to be trained first (run `python src/finetune.py`)

