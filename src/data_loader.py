import pandas as pd
from pathlib import Path


class DataLoader:
    REQUIRED_COLUMNS = ["text"]

    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.data_dir = BASE_DIR / "data"

    def _validate_columns(self, df, dataset_name):
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

    def load_isot(self):
        fake_path = self.data_dir / "isot" / "fake.csv"
        true_path = self.data_dir / "isot" / "true.csv"

        try:
            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset file not found. Check path: {e.filename}"
            )

        # Validate schema
        self._validate_columns(fake_df, "ISOT Fake")
        self._validate_columns(true_df, "ISOT True")

        fake_df["label"] = 1
        true_df["label"] = 0

        fake_df["source_dataset"] = "isot"
        true_df["source_dataset"] = "isot"

        df = pd.concat([fake_df, true_df], ignore_index=True)
        return df

    def unify_datasets(self):
        df = self.load_isot()
        df = df[["text", "label", "source_dataset"]]
        return df