"""
src/models/roberta_model.py

"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -------------------------------------------------------
# Classifier
# -------------------------------------------------------
class RobertaClassifier:
    """
    Fine-tuned RoBERTa binary classifier (fake=1 / real=0).

    API mirrors BertClassifier exactly:
        fit(texts, labels, epochs, batch_size)
        predict_proba(texts)  -> np.ndarray shape (n,)
        predict(texts)        -> np.ndarray shape (n,) of 0/1
        save(path)
        load(path)
    """

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model     = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )
        self.model.to(self.device)

    # --------------------------------------------------
    def fit(self, texts, labels, epochs=2, batch_size=16):
        dataset  = TextDataset(texts, labels, self.tokenizer)
        loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
                optimizer.zero_grad()

                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_tensor  = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor,
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1} loss: {total_loss / len(loader):.4f}")

    # --------------------------------------------------
    def predict_proba(self, texts, batch_size=64):
        self.model.eval()

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="RoBERTa inference"):
                input_ids      = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                p = torch.softmax(outputs.logits, dim=1)
                probs.extend(p.cpu().numpy())

        probs = np.array(probs)
        return probs[:, 1]          # probability of class 1 (fake)

    # --------------------------------------------------
    def predict(self, texts):
        probs = self.predict_proba(texts)
        return (probs >= 0.5).astype(int)

    # --------------------------------------------------
    def save(self, path="models/roberta_finetuned"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"RoBERTa model saved to {path}")

    # --------------------------------------------------
    def load(self, path="models/roberta_finetuned"):
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.model     = RobertaForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)


# -------------------------------------------------------
# Quick weight-check (run directly to verify fine-tuning)
# -------------------------------------------------------
if __name__ == "__main__":
    from transformers import RobertaForSequenceClassification

    model = RobertaForSequenceClassification.from_pretrained("models/roberta_finetuned")
    w = model.classifier.out_proj.weight.data
    print("Classifier weight mean:", w.mean().item())
    print("Classifier weight std: ", w.std().item())
    print("Min:", w.min().item(), "Max:", w.max().item())