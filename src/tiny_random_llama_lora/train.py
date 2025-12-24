"""Training script for creating a LoRA adapter on a tiny random LLaMA model."""

import argparse

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

BASE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def get_tokenizer() -> LlamaTokenizerFast:
    """Load and configure the tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model() -> PeftModelForCausalLM:
    """Load the base model with LoRA configuration."""

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(tokenizer: LlamaTokenizerFast) -> DatasetDict:
    """Load and tokenize the tiny-imdb dataset (only the text column)."""

    dataset = load_dataset("iamholmes/tiny-imdb")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


def train(output_dir: str, num_epochs: int) -> None:
    """Run the training loop."""

    tokenizer = get_tokenizer()
    model = get_model()
    dataset = prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=1e-4,
        fp16=False,
        report_to="none",
    )

    # Causal LM, not masked LM.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Saving.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nLoRA adapter saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on tiny-random-LlamaForCausalLM"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save the LoRA adapter",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    args = parser.parse_args()
    train(output_dir=args.output_dir, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
