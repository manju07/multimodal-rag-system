# Hugging Face Tutorial: Complete Guide to Transformers and Model Hub

## Table of Contents
1. [Introduction to Hugging Face](#introduction)
2. [Installation and Setup](#setup)
3. [Using Pre-trained Models](#pretrained-models)
4. [Transformers Library](#transformers)
5. [Datasets Library](#datasets)
6. [Tokenizers](#tokenizers)
7. [Fine-tuning Models](#fine-tuning)
8. [Model Hub and Sharing](#model-hub)
9. [Inference Endpoints](#inference)
10. [Best Practices](#best-practices)

## Introduction to Hugging Face {#introduction}

Hugging Face is the leading platform for machine learning models, providing:
- **🤗 Transformers**: State-of-the-art ML models for PyTorch, TensorFlow, and JAX
- **🤗 Datasets**: The largest collection of ready-to-use datasets
- **🤗 Model Hub**: Over 300,000+ models shared by the community
- **🤗 Spaces**: Collaborative platform for ML demos and applications

### Key Ecosystems:
- **NLP**: BERT, GPT, RoBERTa, T5, and more
- **Computer Vision**: Vision Transformer, CLIP, DETR
- **Audio**: Wav2Vec2, Whisper, SpeechT5
- **Multimodal**: CLIP, DALL-E, Flamingo
- **Reinforcement Learning**: Decision Transformers

## Installation and Setup {#setup}

### Core Installation
```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers[torch]

# With TensorFlow
pip install transformers[tf]

# Full installation with all dependencies
pip install transformers[all]

# Additional libraries
pip install datasets
pip install tokenizers
pip install accelerate
pip install peft  # For parameter-efficient fine-tuning
pip install bitsandbytes  # For quantization
```

### Environment Setup
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, load_dataset
import numpy as np

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Hugging Face Hub Authentication
```python
from huggingface_hub import login, HfApi
import os

# Method 1: Login interactively
login()

# Method 2: Use token from environment
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"

# Method 3: Programmatic login
login(token="your_token_here")

# Check login status
api = HfApi()
user = api.whoami()
print(f"Logged in as: {user['name']}")
```

## Using Pre-trained Models {#pretrained-models}

### Quick Start with Pipelines
```python
# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love Hugging Face!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Text Generation
generator = pipeline("text-generation", model="gpt2")
output = generator("The future of AI is", max_length=50, num_return_sequences=1)
print(output[0]['generated_text'])

# Question Answering
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company that democratizes AI through open-source and open science."
question = "What does Hugging Face do?"
answer = qa_pipeline(question=question, context=context)
print(answer)

# Named Entity Recognition
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
text = "My name is Sarah and I work at Google in California."
entities = ner_pipeline(text)
print(entities)

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
french_text = translator("Hello, how are you?")
print(french_text)
```

### Available Pipeline Tasks
```python
# Get all available tasks
from transformers import PIPELINE_REGISTRY
print("Available tasks:", list(PIPELINE_REGISTRY.supported_tasks.keys()))

# Specific pipeline examples
pipelines_examples = {
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "token-classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "question-answering": "distilbert-base-cased-distilled-squad",
    "fill-mask": "bert-base-uncased",
    "summarization": "facebook/bart-large-cnn",
    "translation": "t5-base",
    "text-generation": "gpt2",
    "text2text-generation": "t5-small",
    "zero-shot-classification": "facebook/bart-large-mnli",
    "image-classification": "google/vit-base-patch16-224",
    "object-detection": "facebook/detr-resnet-50",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "automatic-speech-recognition": "facebook/wav2vec2-base-960h",
    "text-to-speech": "microsoft/speecht5_tts"
}

# Use any pipeline
for task, model_name in pipelines_examples.items():
    try:
        pipe = pipeline(task, model=model_name)
        print(f"✓ {task}: {model_name}")
    except Exception as e:
        print(f"✗ {task}: {e}")
```

## Transformers Library {#transformers}

### Loading Models and Tokenizers
```python
# Method 1: Auto classes (recommended)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Method 2: Specific model classes
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load model for specific tasks
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Load with specific configurations
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = True
model = AutoModel.from_pretrained(model_name, config=config)
```

### Text Processing
```python
def process_text_with_bert(text, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings
    last_hidden_states = outputs.last_hidden_state
    pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "last_hidden_states": last_hidden_states,
        "pooled_output": pooled_output,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    }

# Usage
text = "Hugging Face transformers are amazing!"
result = process_text_with_bert(text)
print(f"Sequence length: {result['last_hidden_states'].shape[1]}")
print(f"Hidden size: {result['last_hidden_states'].shape[2]}")
```

### Batch Processing
```python
def batch_encode_texts(texts, model_name="bert-base-uncased", batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling for sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0)

# Usage
texts = [
    "I love machine learning",
    "Natural language processing is fascinating",
    "Transformers revolutionized NLP",
    "BERT is a powerful model"
]

embeddings = batch_encode_texts(texts)
print(f"Embeddings shape: {embeddings.shape}")
```

### Model Comparison
```python
def compare_models(text, models):
    results = {}
    
    for model_name in models:
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Process text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            results[model_name] = {
                "embedding_dim": embedding.shape[1],
                "vocab_size": len(tokenizer),
                "max_position": tokenizer.model_max_length,
                "embedding_norm": torch.norm(embedding).item()
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results

# Compare different models
models_to_compare = [
    "bert-base-uncased",
    "roberta-base", 
    "distilbert-base-uncased",
    "albert-base-v2"
]

comparison = compare_models("This is a test sentence.", models_to_compare)
for model, stats in comparison.items():
    print(f"{model}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
```

## Datasets Library {#datasets}

### Loading Datasets
```python
from datasets import load_dataset, Dataset, DatasetDict

# Load popular datasets
imdb = load_dataset("imdb")
squad = load_dataset("squad")
glue_sst2 = load_dataset("glue", "sst2")

# Load specific splits
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test[:1000]")  # First 1000 examples

# Load from local files
local_dataset = load_dataset("csv", data_files="my_data.csv")
json_dataset = load_dataset("json", data_files="my_data.jsonl")

print(f"IMDB dataset: {imdb}")
print(f"Features: {imdb['train'].features}")
print(f"Number of examples: {len(imdb['train'])}")
```

### Dataset Exploration
```python
def explore_dataset(dataset):
    """Explore dataset characteristics"""
    print(f"Dataset: {dataset}")
    print(f"Splits: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"  Size: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        
        # Show first few examples
        print(f"  First example: {split_data[0]}")
        
        # Show data types and statistics
        for feature_name, feature_type in split_data.features.items():
            print(f"  {feature_name}: {feature_type}")

# Explore IMDB dataset
explore_dataset(imdb)
```

### Data Preprocessing
```python
def preprocess_imdb_data(examples, tokenizer, max_length=512):
    """Preprocess IMDB dataset for BERT"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Apply preprocessing
tokenized_imdb = imdb.map(tokenize_function, batched=True)

# Remove original text column and rename label
tokenized_imdb = tokenized_imdb.remove_columns(["text"])
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")

# Set format for PyTorch
tokenized_imdb.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("Preprocessed dataset:")
print(tokenized_imdb["train"][0])
```

### Custom Dataset Creation
```python
def create_custom_dataset():
    """Create a custom dataset"""
    # Sample data
    data = {
        "text": [
            "I love this movie!",
            "This film is terrible.",
            "Great acting and storyline.",
            "Boring and predictable.",
            "Amazing cinematography!"
        ],
        "label": [1, 0, 1, 0, 1]  # 1=positive, 0=negative
    }
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    
    # Split into train/test
    train_test = dataset.train_test_split(test_size=0.2)
    
    return train_test

# Create and use custom dataset
custom_data = create_custom_dataset()
print(custom_data)

# Add preprocessing
def preprocess_custom(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

custom_data = custom_data.map(preprocess_custom, batched=True)
custom_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

### Data Augmentation
```python
def augment_text_data(dataset, augmentation_factor=2):
    """Simple data augmentation for text"""
    import random
    
    def augment_example(example):
        text = example["text"]
        
        # Simple augmentations
        augmented_texts = [text]  # Original
        
        # Synonym replacement (simplified)
        synonyms = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["terrible", "awful", "horrible"],
            "nice": ["pleasant", "lovely", "delightful"]
        }
        
        for word, syns in synonyms.items():
            if word in text.lower():
                for syn in syns:
                    augmented_texts.append(text.lower().replace(word, syn))
        
        # Return multiple examples
        return {
            "text": augmented_texts[:augmentation_factor],
            "label": [example["label"]] * min(len(augmented_texts), augmentation_factor)
        }
    
    # Apply augmentation
    augmented = dataset.map(augment_example, remove_columns=dataset.column_names, batched=False)
    
    return augmented
```

## Tokenizers {#tokenizers}

### Understanding Tokenization
```python
from transformers import AutoTokenizer

def analyze_tokenization(text, model_names):
    """Analyze how different tokenizers handle the same text"""
    print(f"Original text: '{text}'\n")
    
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"Model: {model_name}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Number of tokens: {len(tokens)}")
        print()

# Compare different tokenizers
text = "Hello, I'm using Hugging Face transformers!"
models = [
    "bert-base-uncased",
    "gpt2",
    "roberta-base",
    "t5-base"
]

analyze_tokenization(text, models)
```

### Custom Tokenization
```python
def custom_tokenization_pipeline(texts, model_name="bert-base-uncased"):
    """Custom tokenization with special handling"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = []
    for text in texts:
        # Basic tokenization
        basic_tokens = tokenizer(text)
        
        # Add special tokens handling
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True if "bert" in model_name else False
        )
        
        # Token analysis
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        
        results.append({
            "original_text": text,
            "tokens": tokens,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens": {
                "CLS": tokenizer.cls_token,
                "SEP": tokenizer.sep_token,
                "PAD": tokenizer.pad_token,
                "UNK": tokenizer.unk_token
            }
        })
    
    return results

# Usage
texts = [
    "Short text",
    "This is a much longer text that might need truncation depending on the model's maximum sequence length",
    "Text with special characters: @#$%!"
]

tokenization_results = custom_tokenization_pipeline(texts)
for i, result in enumerate(tokenization_results):
    print(f"Example {i+1}:")
    print(f"  Original: {result['original_text']}")
    print(f"  Tokens: {result['tokens'][:10]}...")  # First 10 tokens
    print(f"  Length: {len(result['tokens'])}")
    print()
```

### Fast Tokenizers
```python
from transformers import AutoTokenizer
import time

def compare_tokenizer_speed(texts, model_name="bert-base-uncased"):
    """Compare fast vs slow tokenizer performance"""
    
    # Load both versions
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    slow_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Benchmark fast tokenizer
    start_time = time.time()
    fast_results = fast_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    fast_time = time.time() - start_time
    
    # Benchmark slow tokenizer  
    start_time = time.time()
    slow_results = slow_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    slow_time = time.time() - start_time
    
    print(f"Fast tokenizer: {fast_time:.4f} seconds")
    print(f"Slow tokenizer: {slow_time:.4f} seconds")
    print(f"Speedup: {slow_time/fast_time:.2f}x")
    
    return fast_results, slow_results

# Test with many texts
large_texts = ["This is a test sentence."] * 1000
fast_result, slow_result = compare_tokenizer_speed(large_texts)
```

## Fine-tuning Models {#fine-tuning}

### Basic Fine-tuning Setup
```python
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def setup_fine_tuning(model_name, num_labels):
    """Setup model and tokenizer for fine-tuning"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### Text Classification Fine-tuning
```python
def fine_tune_text_classifier():
    """Fine-tune BERT for text classification"""
    
    # Load and preprocess data
    dataset = load_dataset("imdb")
    model, tokenizer = setup_fine_tuning("bert-base-uncased", num_labels=2)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    # Preprocess datasets
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.remove_columns(["text"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch")
    
    # Use smaller subset for demo
    train_dataset = encoded_dataset["train"].shuffle().select(range(1000))
    eval_dataset = encoded_dataset["test"].shuffle().select(range(200))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model, tokenizer, trainer

# Run fine-tuning (commented out for demo)
# model, tokenizer, trainer = fine_tune_text_classifier()
```

### Parameter-Efficient Fine-tuning (LoRA)
```python
from peft import get_peft_model, LoraConfig, TaskType

def setup_lora_fine_tuning(model_name, num_labels):
    """Setup LoRA fine-tuning for efficient training"""
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        inference_mode=False,
        r=8,                        # Rank
        lora_alpha=32,              # LoRA scaling parameter
        lora_dropout=0.1,           # LoRA dropout
        target_modules=["query", "value"],  # Target attention modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

# Usage
lora_model = setup_lora_fine_tuning("bert-base-uncased", num_labels=2)
```

### Custom Training Loop
```python
def custom_training_loop(model, tokenizer, train_dataloader, eval_dataloader, num_epochs=3):
    """Custom training loop with more control"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        eval_accuracy = evaluate_model(model, eval_dataloader)
        print(f"Evaluation accuracy: {eval_accuracy:.4f}")
        model.train()

def evaluate_model(model, dataloader):
    """Evaluate model accuracy"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    return correct / total
```

## Model Hub and Sharing {#model-hub}

### Exploring the Hub
```python
from huggingface_hub import HfApi, list_models, model_info

api = HfApi()

# List models with filters
models = list_models(
    task="text-classification",
    library="transformers",
    language="en",
    sort="downloads",
    limit=10
)

print("Top 10 English text classification models:")
for model in models:
    print(f"- {model.id} ({model.downloads} downloads)")

# Get detailed model information
model_details = model_info("bert-base-uncased")
print(f"\nModel: {model_details.id}")
print(f"Library: {model_details.library_name}")
print(f"Downloads: {model_details.downloads}")
print(f"Tags: {model_details.tags}")
```

### Uploading Models
```python
from huggingface_hub import Repository, upload_folder

def upload_model_to_hub(model, tokenizer, repo_name, commit_message="Upload model"):
    """Upload trained model to Hugging Face Hub"""
    
    # Save model locally first
    model.save_pretrained(f"./{repo_name}")
    tokenizer.save_pretrained(f"./{repo_name}")
    
    # Create model card
    model_card_content = f"""
---
language: en
license: apache-2.0
tags:
- text-classification
- sentiment-analysis
datasets:
- imdb
metrics:
- accuracy
---

# {repo_name}

This model is a fine-tuned version of BERT for sentiment analysis on the IMDB dataset.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("your-username/{repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("your-username/{repo_name}")

# Use the model
inputs = tokenizer("I love this movie!", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Training Data

The model was trained on the IMDB movie reviews dataset.

## Training Procedure

- Learning rate: 2e-5
- Batch size: 16
- Number of epochs: 3

## Evaluation Results

- Accuracy: 92.5%
- F1-score: 0.925
"""
    
    with open(f"./{repo_name}/README.md", "w") as f:
        f.write(model_card_content)
    
    # Upload to hub
    upload_folder(
        folder_path=f"./{repo_name}",
        repo_id=f"your-username/{repo_name}",
        repo_type="model",
        commit_message=commit_message
    )

# Usage (example)
# upload_model_to_hub(fine_tuned_model, tokenizer, "my-sentiment-model")
```

### Model Versioning and Management
```python
def manage_model_versions(repo_id):
    """Manage different versions of a model"""
    api = HfApi()
    
    # List all commits (versions)
    commits = api.list_repo_commits(repo_id)
    print(f"Model versions for {repo_id}:")
    for commit in commits[:5]:  # Show last 5 versions
        print(f"- {commit.commit_id[:8]}: {commit.title}")
    
    # Load specific version
    specific_version_model = AutoModel.from_pretrained(
        repo_id,
        revision=commits[1].commit_id  # Load second-to-last version
    )
    
    return specific_version_model

# Usage
# model_versions = manage_model_versions("bert-base-uncased")
```

## Inference Endpoints {#inference}

### Local Inference Optimization
```python
import torch
from transformers import pipeline

def optimize_for_inference(model_name):
    """Optimize model for faster inference"""
    
    # Load with optimizations
    classifier = pipeline(
        "text-classification",
        model=model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto"           # Automatic device mapping
    )
    
    # Batch processing function
    def batch_predict(texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = classifier(batch)
            results.extend(batch_results)
        return results
    
    return classifier, batch_predict

# Usage
classifier, batch_predict = optimize_for_inference("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Test batch prediction
texts = ["I love this!", "This is terrible", "Not bad"] * 100
results = batch_predict(texts)
print(f"Processed {len(results)} texts")
```

### Quantization for Edge Deployment
```python
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

def quantize_model(model_name, quantization_type="8bit"):
    """Quantize model for reduced memory usage"""
    
    if quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    return model

# Usage
quantized_model = quantize_model("bert-base-uncased", "8bit")
print(f"Model memory footprint reduced with 8-bit quantization")
```

### ONNX Export for Production
```python
def export_to_onnx(model_name, output_path="model.onnx"):
    """Export model to ONNX format for production deployment"""
    from transformers.onnx import export
    from transformers import AutoTokenizer, AutoModel
    from pathlib import Path
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Export to ONNX
    onnx_path = Path(output_path)
    export(
        preprocessor=tokenizer,
        model=model,
        config=model.config,
        opset=14,
        output=onnx_path
    )
    
    return onnx_path

# Usage
# onnx_path = export_to_onnx("distilbert-base-uncased")
# print(f"Model exported to {onnx_path}")
```

## Best Practices {#best-practices}

### Memory Management
```python
import gc
import torch

def optimize_memory_usage():
    """Best practices for memory management"""
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Use gradient checkpointing for large models
    model.gradient_checkpointing_enable()
    
    # Use mixed precision training
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    # Training with mixed precision
    with autocast():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU memory free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
```

### Error Handling and Robustness
```python
def robust_model_loading(model_name, fallback_model="distilbert-base-uncased"):
    """Robust model loading with fallback"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"Successfully loaded {model_name}")
        return model, tokenizer
    
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print(f"Falling back to {fallback_model}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModel.from_pretrained(fallback_model)
            return model, tokenizer
        except Exception as e:
            print(f"Failed to load fallback model: {e}")
            raise

def safe_inference(model, tokenizer, text, max_retries=3):
    """Safe inference with retry logic"""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            return outputs
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error, attempt {attempt + 1}/{max_retries}")
                torch.cuda.empty_cache()
                # Reduce batch size or sequence length
                if attempt < max_retries - 1:
                    continue
            raise
        except Exception as e:
            print(f"Inference error: {e}")
            if attempt == max_retries - 1:
                raise
```

### Performance Monitoring
```python
import time
from functools import wraps

def benchmark_function(func):
    """Decorator to benchmark function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@benchmark_function
def benchmark_model_inference(model, tokenizer, texts):
    """Benchmark model inference speed"""
    all_results = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        all_results.append(outputs)
    
    return all_results

# Usage
texts = ["Sample text for benchmarking"] * 100
# results = benchmark_model_inference(model, tokenizer, texts)
```

### Model Selection Guide
```python
def recommend_model(task, performance_priority="balanced"):
    """Recommend model based on task and performance requirements"""
    
    recommendations = {
        "text-classification": {
            "speed": "distilbert-base-uncased",
            "balanced": "bert-base-uncased", 
            "accuracy": "roberta-large"
        },
        "question-answering": {
            "speed": "distilbert-base-cased-distilled-squad",
            "balanced": "bert-base-cased",
            "accuracy": "roberta-large-squad2"
        },
        "text-generation": {
            "speed": "gpt2",
            "balanced": "gpt2-medium",
            "accuracy": "gpt2-large"
        },
        "summarization": {
            "speed": "facebook/bart-base",
            "balanced": "facebook/bart-large-cnn",
            "accuracy": "google/pegasus-large"
        }
    }
    
    if task in recommendations:
        return recommendations[task].get(performance_priority, recommendations[task]["balanced"])
    else:
        return "bert-base-uncased"  # Default fallback

# Usage
model_name = recommend_model("text-classification", "speed")
print(f"Recommended model: {model_name}")
```

## Conclusion

Hugging Face provides a comprehensive ecosystem for working with transformer models. This tutorial covered:

- Using pre-trained models with pipelines
- Understanding the Transformers library architecture
- Working with datasets and tokenizers
- Fine-tuning models for custom tasks
- Sharing models on the Hub
- Optimizing models for production

### Key Takeaways:
1. Start with pipelines for quick prototyping
2. Use Auto classes for flexibility
3. Preprocess data carefully for best results
4. Consider parameter-efficient fine-tuning (LoRA/QLoRA)
5. Optimize models for production deployment
6. Monitor performance and memory usage

### Next Steps:
1. Explore domain-specific models on the Hub
2. Experiment with multimodal models (vision + language)
3. Try advanced fine-tuning techniques
4. Build end-to-end applications with Gradio/Streamlit
5. Contribute models back to the community

### Additional Resources:
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Course](https://huggingface.co/course)
- [Community Forums](https://discuss.huggingface.co)
- [Model Hub](https://huggingface.co/models)
