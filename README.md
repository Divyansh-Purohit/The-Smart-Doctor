# The Smart Doctor

An end-to-end medical AI pipeline for skin cancer detection and clinical consultation. The project covers four areas: vision-based lesion classification, model explainability, LLM fine-tuning, and retrieval-augmented generation.

**Part 1 — Vision Diagnostics**
Fine-tunes ResNet-18 and ViT-Base-16 on the DermaMNIST dataset (7-class skin lesion classification). Compares accuracy, training dynamics, and confusion matrices for both architectures.

**Part 2 — Explainability**
Applies GradCAM on ResNet-18 and Attention Rollout on ViT to visualise what each model focuses on. Analyses five diagnostic cases: both correct, both wrong, and cases where one model outperforms the other.

**Part 3 — Medical LLM with LoRA**
Fine-tunes Gemma-3-270M-IT on the MedMCQA dataset (medical entrance exam questions) using LoRA. Compares zero-shot baseline accuracy vs. the fine-tuned model on a 100-sample eval set.

**Part 4 — RAG Pipeline**
Builds a retrieval-augmented generation pipeline with LlamaIndex, BAAI/bge-small-en-v1.5 embeddings, and Gemma3:4b via Ollama. Retrieves relevant chunks from medical research papers before generation. Includes a chunk-size hyperparameter sweep and a RAG vs. no-RAG comparison.

## Setup

Requires Python 3.10+. Install dependencies:

```bash
uv sync
uv add opencv-python medmnist transformers peft trl datasets wandb huggingface_hub \
       llama-index llama-index-llms-ollama llama-index-embeddings-huggingface \
       sentence-transformers
```

For Part 4, install and start Ollama with the Gemma3 4B model:

```bash
brew install ollama
ollama pull gemma3:4b
```

For Part 3, set your Hugging Face token (required to access Gemma):

```bash
export HF_TOKEN=hf_...
```

## Running the notebook

```bash
uv run jupyter notebook The_Smart_Doctor.ipynb
```

The notebook has `skip_training = True` by default. All trained weights and eval results are in `outputs/` so the notebook runs end-to-end without retraining. Set `skip_training = False` to retrain from scratch.

> **Note:** `outputs/vit_b16_best.pth` (327MB) is excluded from the repo. If it is missing, the ViT sections will retrain automatically when `skip_training = False`.

## Project structure

```
The_Smart_Doctor.ipynb   main notebook
documents/               medical research papers used by the RAG pipeline
outputs/
  resnet18_best.pth      trained ResNet-18 weights
  gemma_lora/            LoRA adapter for Gemma-3-270M-IT
  rag_index/             persisted vector index
  gemma_results.json     baseline evaluation results
  peft_gemma_results.json fine-tuned evaluation results
  *.png                  training curves, confusion matrices, heatmaps
requirements.txt
```
