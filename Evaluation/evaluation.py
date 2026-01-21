import os
import random
from os.path import join

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'
h_stop = '\x1b[0m'

multicw_path = join('Final-dataset')

# -----------------------------
# Dataset helper
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -----------------------------
# Base model class
# -----------------------------
class HFTextClassifier:
    def __init__(self, model_name, num_labels=2, max_len=256, batch_size=32):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            return False
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"{self.model_name} loaded from {model_path}")
        return True

    def train_model(self, train_df, dev_df, model_save_name, epochs=5, learn_rate=2e-6):
        train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), self.tokenizer, self.max_len)
        dev_dataset = TextDataset(dev_df["text"].tolist(), dev_df["label"].tolist(), self.tokenizer, self.max_len)

        training_args = TrainingArguments(
            output_dir=f"Models/{model_save_name}",
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            learning_rate=learn_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=200,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        trainer.train()
        trainer.save_model(f"Models/{model_save_name}")
        self.tokenizer.save_pretrained(f"Models/{model_save_name}")
        print(f"{self.model_name} trained and saved to Models/{model_save_name}")

    def evaluate(self, test_df):
        dataset = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(), self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        report_dict = classification_report(test_df["label"].to_numpy(), np.array(preds), output_dict=True)
        report_str = classification_report(test_df["label"].to_numpy(), np.array(preds))
        return report_dict, report_str

# -----------------------------
# Model wrappers
# -----------------------------
class XLMRobertaClassifier(HFTextClassifier):
    def __init__(self, num_labels=2):
        super().__init__("xlm-roberta-base", num_labels=num_labels)

class mDeBERTaClassifier(HFTextClassifier):
    def __init__(self, num_labels=2):
        super().__init__("microsoft/mdeberta-v3-base", num_labels=num_labels)


if __name__ == "__main__":
    import pandas as pd

    # Load data with better type handling
    multicw_train = pd.read_csv(join(multicw_path, "multicw-train.csv"))
    multicw_dev = pd.read_csv(join(multicw_path, "multicw-dev.csv"))
    multicw_test = pd.read_csv(join(multicw_path, "multicw-test.csv"))

    # Clean and convert labels - handle any NaN values
    multicw_train['label'] = pd.to_numeric(multicw_train['label'], errors='coerce').fillna(0).astype(np.int32)
    multicw_dev['label'] = pd.to_numeric(multicw_dev['label'], errors='coerce').fillna(0).astype(np.int32)
    multicw_test['label'] = pd.to_numeric(multicw_test['label'], errors='coerce').fillna(0).astype(np.int32)

    # Ensure text columns are strings and handle NaN
    multicw_train['text'] = multicw_train['text'].fillna('').astype(str)
    multicw_dev['text'] = multicw_dev['text'].fillna('').astype(str)
    multicw_test['text'] = multicw_test['text'].fillna('').astype(str)

    # Remove any rows with empty text
    multicw_train = multicw_train[multicw_train['text'].str.strip() != '']
    multicw_dev = multicw_dev[multicw_dev['text'].str.strip() != '']
    multicw_test = multicw_test[multicw_test['text'].str.strip() != '']

    print(f'Loaded MultiCW:')
    print(f'Train set: {multicw_train.shape[0]}')
    print(f'Dev set: {multicw_dev.shape[0]}')
    print(f'Test set: {multicw_test.shape[0]}')

    # Training loop
    models = ['xlm', 'mdb']
    seeds = [42, 123, 456]

    for model in models:
        for seed in seeds:
            print(f'\n{h_green}Model: {model} | Seed: {seed}{h_stop}')

            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Initialize model
            detector = None
            if model == 'xlm':
                detector = XLMRobertaClassifier()
            elif model == 'mdb':
                detector = mDeBERTaClassifier()

            model_name_seeded = f'{model}-multicw-seed{seed}'

            # Load model or train
            if not detector.load_model(model_name_seeded):
                print(f'{h_yellow}No model found. Initiating fine-tuning:{h_stop}')

                # Make sure DataFrames are reset and clean before passing
                train_clean = multicw_train.reset_index(drop=True).copy()
                dev_clean = multicw_dev.reset_index(drop=True).copy()

                detector.train_model(
                    train_clean,
                    dev_clean,
                    epochs=5,
                    learn_rate=2e-6,
                    model_save_name=model_name_seeded,
                )

            # Evaluate - Changed from detect_claims to evaluate
            print(f'{h_yellow}MultiCW overall:{h_stop}')
            _, report = detector.evaluate(multicw_test)
            print(report)

            # Check if 'style' column exists before filtering
            if 'style' in multicw_test.columns:
                test_noisy = multicw_test.loc[multicw_test['style'] == 'noisy'].reset_index(drop=True)
                if len(test_noisy) > 0:
                    _, report = detector.evaluate(test_noisy)
                    print(f'{h_yellow}MultiCW Noisy Part:{h_stop}')
                    print(report)

                test_strut = multicw_test.loc[multicw_test['style'] == 'struct'].reset_index(drop=True)
                if len(test_strut) > 0:
                    _, report = detector.evaluate(test_strut)
                    print(f'{h_yellow}MultiCW Structured Part:{h_stop}')
                    print(report)
            else:
                print(f'{h_yellow}Note: "style" column not found in test data. Skipping subset evaluations.{h_stop}')