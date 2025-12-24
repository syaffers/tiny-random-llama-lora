# Tiny Random LLaMA LoRA

A minimal LoRA finetuning code for [hmellor/tiny-random-LlamaForCausalLM](https://huggingface.co/hmellor/tiny-random-LlamaForCausalLM). This LoRA adapter is only useful for smoke testing deployments.

You can find the LoRA adapter on HuggingFace at [syaffers/tiny-random-llama-lora](https://huggingface.co/syaffers/tiny-random-llama-lora).

## Overview

This project creates a LoRA adapter by finetuning on the [iamholmes/tiny-imdb](https://huggingface.co/datasets/iamholmes/tiny-imdb) dataset. I just wanted to keep things inside the HuggingFace ecosystem and maintain model hierarchies.

## Installation

Make sure you have `uv` installed. Then, simply

```bash
uv sync
```

## Usage

### Train the LoRA Adapter

```bash
uv run train-lora --output-dir ./outputs --num-epochs 3
```

### Upload to HuggingFace Hub

First, log in to HuggingFace:

```bash
uv run hf auth login
```

You will be prompted to fill in your access token. Then upload your adapter:

```bash
uv run upload-lora --adapter-path ./outputs --repo-id <your username>/tiny-random-llama-lora
```

## Configuration

The LoRA configuration uses:
- **r**: 8 (rank)
- **lora_alpha**: 16
- **target_modules**: `["q_proj", "v_proj"]`
- **lora_dropout**: 0.05

## License

MIT
