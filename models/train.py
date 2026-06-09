"""
DriftShield Model Training Module.

Finetunes BioBERT for sequence classification with evaluation, checkpointing, and experiment tracking via Weights & Biases.
"""

import json
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env) so WANDB_API_KEY is available
load_dotenv()
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import torch
import datasets
import wandb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    PreTrainedTokenizer, PreTrainedTokenizerFast
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

@dataclass
class TrainConfig:
    """Configuration class for DriftShield model training.

    Attributes:
        model_name: Base encoder checkpoint model name.
        max_length: Maximum sequence length.
        num_epochs: Number of training epochs.
        train_batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 weight decay regularization.
        warmup_ratio: Linear warmup ratio.
        dropout: Classifier head dropout value.
        seed: Random seed.
        fp16: Enable float16 training.
        bf16: Enable bfloat16 training.
        early_stopping_patience: Patience epochs for early stopping.
        output_dir: Checkpoints directory path.
        wandb_project: Weights & Biases project name.
        data_dir: Source folder for processed datasets.
    """
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    max_length: int = 256
    num_epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.3
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    early_stopping_patience: int = 2
    output_dir: str = "checkpoints"
    wandb_project: str = "driftshield"
    data_dir: str = "data/processed"


def load_hf_dataset(
    data_dir: Path, 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    max_length: int
) -> datasets.DatasetDict:
    """Loads JSON datasets and tokenizes them into Hugging Face Dataset format.

    Args:
        data_dir: Path to the directory containing JSON splits.
        tokenizer: PreTrained HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Hugging Face DatasetDict containing train, val, and test splits.
    """
    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            batch["context"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    splits = {}
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = datasets.Dataset.from_list(data)
        ds = ds.map(tokenize, batched=True)
        ds = ds.rename_column("label", "labels")
        # Ensure we set appropriate PyTorch format columns
        ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
        splits[split] = ds
    return datasets.DatasetDict(splits)


def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """Computes evaluation classification metrics.

    Args:
        eval_pred: A tuple of (logits, labels).

    Returns:
        Dict mapping metric names to computed values.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
    }


def train(config: TrainConfig) -> None:
    """Trains the sequence classification model.

    Args:
        config: TrainConfig instance holding model and hyperparameter values.
    """
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    wandb.init(project=config.wandb_project, config=vars(config))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
    )

    dataset = load_hf_dataset(Path(config.data_dir), tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=config.seed,
        fp16=config.fp16 if torch.cuda.is_available() else False,
        bf16=config.bf16 if torch.cuda.is_available() else False,
        report_to="wandb" if os.environ.get("WANDB_MODE") != "disabled" else "none",
        run_name="driftshield-biobert-finetune",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()

    best_dir = Path(config.output_dir) / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    test_results = trainer.evaluate(dataset["test"])
    with open(Path(config.output_dir) / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # Log test results to W&B as a summary table
    if os.environ.get("WANDB_MODE") != "disabled":
        wandb.log({"test/" + k: v for k, v in test_results.items()})
        artifact = wandb.Artifact("test-results", type="evaluation")
        artifact.add_file(str(Path(config.output_dir) / "test_results.json"))
        wandb.log_artifact(artifact)

    print(f"Test Results: {test_results}")
    wandb.finish()


if __name__ == "__main__":
    train(TrainConfig())
