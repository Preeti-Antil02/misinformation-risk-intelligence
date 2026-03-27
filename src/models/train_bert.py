# run: python -m src.models.train_bert

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.models.bert_model import BertClassifier

from sklearn.model_selection import train_test_split


def main():

    print("Loading dataset...")

    loader = DataLoader()
    df = loader.load_combined(sample_welfake=False)

    cleaner = DataCleaner(min_words=3)

    df = cleaner.remove_duplicates(df)
    df = cleaner.remove_nulls(df)
    df = cleaner.remove_short_texts(df)

    print("Preprocessing text...")

    df["text"] = df["text"].astype(str)

    tp = TextPreprocessor()
    df["text"] = df["text"].apply(tp.basic_clean)
    df["text"] = df["text"].apply(tp.truncate)    

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Fine-tuning BERT...")

    bert = BertClassifier()

    bert.fit(
        X_train.tolist(),
        y_train.tolist(),
        epochs=2,       
        batch_size=16
    )
    bert.save("models/bert_finetuned")

    print("BERT fine-tuned and saved.")


if __name__ == "__main__":
    main()