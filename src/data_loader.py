import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.data_dir = BASE_DIR / "data"

    def load_isot(self):
        fake_path = self.data_dir / "isot" / "fake.csv"
        true_path = self.data_dir / "isot" / "true.csv"

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df["label"] = 1
        true_df["label"] = 0

        fake_df["source_dataset"] = "isot"
        true_df["source_dataset"] = "isot"

        df = pd.concat([fake_df, true_df], ignore_index=True)
        return df

    def unify_datasets(self):
        isot_df = self.load_isot()
        df = isot_df
        df = df[["text", "label", "source_dataset"]]
        return df