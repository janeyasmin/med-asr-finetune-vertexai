import argparse
import multiprocessing
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import subprocess, sys

import jiwer
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# This command lists the installed Python packages. It's useful for debugging environment issues.
subprocess.run([sys.executable, "-m", "pip" "list"])


def train(args):
    # Prepare dataset
    MODEL_ID = "google/medasr"
    SAMPLING_RATE = 16000

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def prepare_dataset(batch):
        """Preprocesses a batch of raw dataset examples for CTC training."""
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=SAMPLING_RATE
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    @dataclass
    class DataCollator:
        """Extracts features and pads them within a batch."""

        processor: Any
        padding: Union[bool, str] = True

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
                return_attention_mask=True,
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )
            batch["labels"] = labels_batch["input_ids"]
            return batch

    def normalize_text(text):
        """Normalize text by lowercasing and removing punctuation for WER calculation."""
        if not text:
            return ""
        text = text.lower()
        text = text.replace("</s>", "")
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def compute_metrics(pred):
        """Computes the Word Error Rate (WER) for evaluation."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        pred_str_norm = [normalize_text(s) for s in pred_str]
        label_str_norm = [normalize_text(s) for s in label_str]
        return {"wer": jiwer.wer(label_str_norm, pred_str_norm)}

    raw_dataset = load_dataset(
        "ekacare/eka-medical-asr-evaluation-dataset", name="en", split="test"
    )

    # For a 70/30 fine-tune training / validation split, set test_size = 0.3.
    test_size = 0.3
    random_seed = 42  # setting the seed ensures the training / validation split is identical every time

    min_audio_length_sec = 0.6
    print("*** generating train test split ***")
    dataset_splits = (
        raw_dataset.train_test_split(  # pyright: ignore[reportAttributeAccessIssue]
            test_size=test_size, seed=random_seed
        )
    )
    print("*** extracting audio data from Audio file ***")
    dataset_splits = dataset_splits.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    print("*** filtering audio data by min audio length ***")
    dataset_splits = dataset_splits.filter(
        lambda x: len(x["audio"]["array"]) > min_audio_length_sec * SAMPLING_RATE,
    )
    print("*** preprocessing batch of raw dataset examples for CTC training ***")
    encoded_dataset = dataset_splits.map(
        prepare_dataset,
        remove_columns=dataset_splits["train"].column_names,  # Remove old columns
    )

    # Retrieve model directory path
    model_dir = os.environ.get("AIP_MODEL_DIR")
    print(f"*** extracting AIP_MODEL_DIR: {model_dir} ***")

    # If the output directory is a GCS path, convert it to a local path that FUSE can use.
    if model_dir and model_dir.startswith("gs://"):
        # Convert 'gs://bucket/path' to '/gcs/bucket/path'
        model_dir = model_dir.replace("gs://", "/gcs/", 1)

    # Load the pretrained Whisper model for conditional generation.
    if args.do_train:
        print(f" *** Loading model from HF Hub for training ***")
        model = AutoModelForCTC.from_pretrained(MODEL_ID)
    else:
        print(f" *** Loading model from {model_dir} for evaluation ***")
        model = AutoModelForCTC.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    training_args = TrainingArguments(
        output_dir=model_dir,
        # Performance
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        dataloader_num_workers=max(2, multiprocessing.cpu_count() - 2),
        dataloader_pin_memory=True,
        # Optimization
        num_train_epochs=args.epochs,
        learning_rate=3e-5,
        warmup_steps=300,
        optim="adamw_torch",
        # Evaluation and save model
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Logging
        logging_steps=30,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        processing_class=processor.feature_extractor,
        data_collator=DataCollator(processor=processor),
        compute_metrics=compute_metrics,
    )

    # Start the training process.
    if args.do_train:
        print("*** Starting training ***")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.create_model_card()
        tokenizer.save_pretrained(model_dir)

    if args.do_eval:
        print("*** Starting evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    train(parser.parse_args())
