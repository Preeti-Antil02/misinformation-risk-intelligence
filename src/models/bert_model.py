import encodings
from unittest import loader

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=128):
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
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BertClassifier:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def fit(self, texts, labels, epochs=2, batch_size=8):

        dataset = TextDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):

                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} loss:", total_loss / len(loader))

    def load(self, path="models/bert_finetuned"):

        self.tokenizer = BertTokenizer.from_pretrained(path)

        self.model = BertForSequenceClassification.from_pretrained(path)

        self.model.to(self.device)

    def predict(self, texts):

        probs = self.predict_proba(texts)

        return (probs >= 0.5).astype(int)


    
    def predict_proba(self, texts, batch_size=64):

        self.model.eval()

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"]
        )

        loader = DataLoader(dataset, batch_size=batch_size)

        probs = []

        with torch.no_grad():

            for batch in tqdm(loader, desc="BERT inference"):

                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                p = torch.softmax(outputs.logits, dim=1)

                probs.extend(p.cpu().numpy())

        probs = np.array(probs)
        return probs[:, 1]

    def save(self, path="models/bert_finetuned"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"BERT model saved to {path}")
        



# paste this temporarily at the bottom of bert_model.py and run it directly
if __name__ == "__main__":
    import os
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained("models/bert_finetuned")
    
    # Check classifier weights — if fine-tuned, these should NOT be near zero
    w = model.classifier.weight.data
    print("Classifier weight mean:", w.mean().item())
    print("Classifier weight std: ", w.std().item())
    print("Min:", w.min().item(), "Max:", w.max().item())