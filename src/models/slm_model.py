"""
src/models/slm_model.py

Qwen2.5-3B-Instruct — zero-shot misinformation classifier.
No fine-tuning required. Uses structured prompt + JSON output parsing.

Interface mirrors BertClassifier and RobertaClassifier:
    predict_proba(texts)  -> np.ndarray shape (n,)   float 0-1
    predict(texts)        -> np.ndarray shape (n,)   int 0/1
    load(path)            -> loads from local path (optional)
    save(path)            -> not applicable for zero-shot

Why Qwen2.5-3B:
  - ~110 tokens/sec on CPU  (vs Phi-4-mini 43 t/s)
  - Strong instruction following for structured JSON output
  - Apache 2.0 license
  - 3B params sufficient for zero-shot fake news reasoning
  - Better cross-domain than fine-tuned BERT (no style bias)
"""

import json
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------------------------------
# Prompt template
# -------------------------------------------------------
SYSTEM_PROMPT = """You are an expert misinformation analyst. Your job is to assess whether a news article or claim is likely fake or real.

Analyse the text for these signals:
- Sensational or clickbait language
- Extraordinary claims without credible sources
- Emotional manipulation tactics
- Conspiracy framing ("they don't want you to know")
- Urgency language ("share before deleted")
- Scientific implausibility
- Lack of named sources or institutions

Respond with ONLY a valid JSON object, no other text:
{"label": "fake" or "real", "confidence": 0.0 to 1.0, "reason": "one sentence explanation"}"""

USER_TEMPLATE = "Analyse this article:\n\n{text}"


# -------------------------------------------------------
# Classifier
# -------------------------------------------------------
class QwenClassifier:
    """
    Zero-shot misinformation classifier using Qwen2.5-3B-Instruct.

    API mirrors BertClassifier / RobertaClassifier:
        predict_proba(texts, batch_size) -> np.ndarray (n,)
        predict(texts)                  -> np.ndarray (n,) of 0/1
        load(path)                      -> load from local HF path
    """

    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or self.MODEL_ID
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model     = None

    # --------------------------------------------------
    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self.model is None:
            print(f"Loading {self.model_id} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            print("Qwen2.5-3B-Instruct loaded.")

    # --------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        """Build chat-formatted prompt for Qwen."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(text=text[:1000])},
        ]
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback manual format
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{USER_TEMPLATE.format(text=text[:1000])}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    # --------------------------------------------------
    def _parse_response(self, raw: str) -> float:
        """
        Parse model output to a fake probability float.
        Returns 0.5 (uncertain) if parsing fails.
        """
        # Extract JSON block
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not match:
            return 0.5

        try:
            data = json.loads(match.group())
            label      = str(data.get("label", "")).lower().strip()
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            if label == "fake":
                return confidence
            elif label == "real":
                return 1.0 - confidence
            else:
                return 0.5

        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: check if raw text contains "fake" or "real"
            raw_lower = raw.lower()
            if "fake" in raw_lower and "real" not in raw_lower:
                return 0.75
            elif "real" in raw_lower and "fake" not in raw_lower:
                return 0.25
            return 0.5

    # --------------------------------------------------
    def predict_proba(self, texts: list, batch_size: int = 1) -> np.ndarray:
        """
        Returns array of fake probabilities, shape (n,).
        batch_size=1 is default — generative models are best run one at a time
        to allow proper JSON generation per sample.
        """
        self._ensure_loaded()

        probs = []

        for text in tqdm(texts, desc="Qwen inference"):
            prompt = self._build_prompt(str(text))

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_ids   = output_ids[0][input_len:]
            raw       = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            prob = self._parse_response(raw)
            probs.append(prob)

        return np.array(probs, dtype=np.float32)

    # --------------------------------------------------
    def predict(self, texts: list) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs >= 0.5).astype(int)

    # --------------------------------------------------
    def load(self, path: str):
        """Load from a local directory (if model was cached/downloaded)."""
        print(f"Loading Qwen from local path: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def save(self, path: str):
        """Cache downloaded model locally for faster future loads."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call predict_proba() first.")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Qwen model cached to {path}")


# -------------------------------------------------------
# Quick test
# -------------------------------------------------------
if __name__ == "__main__":
    clf = QwenClassifier()

    test_texts = [
        "Scientists confirm drinking hot water cures cancer. Big pharma is hiding this.",
        "The Federal Reserve held interest rates steady, citing sustained inflation data.",
        "NASA discovers hidden alien city on Mars but suppresses images to avoid panic.",
        "Parliament passed a new data privacy bill with cross-party support on Tuesday.",
    ]

    probs = clf.predict_proba(test_texts)
    preds = clf.predict(test_texts)

    for text, prob, pred in zip(test_texts, probs, preds):
        label = "FAKE" if pred == 1 else "REAL"
        print(f"[{label}] {prob:.3f} — {text[:60]}...")