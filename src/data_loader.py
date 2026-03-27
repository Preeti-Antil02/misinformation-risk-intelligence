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

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        # Validate title column
        if "title" not in fake_df.columns or "title" not in true_df.columns:
            raise ValueError("ISOT dataset must contain a 'title' column")

        # Use title instead of full article text
        fake_df["text"] = fake_df["title"]
        true_df["text"] = true_df["title"]

        fake_df["label"] = 1
        true_df["label"] = 0

        fake_df["source_dataset"] = "isot"
        true_df["source_dataset"] = "isot"

        df = pd.concat([fake_df, true_df], ignore_index=True)

        return df[["text", "label", "source_dataset"]]

    def load_welfake(self):

        path = self.data_dir / "welfake" / "WELFake_Dataset.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"WELFake dataset not found at {path}. "
                f"Download from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification "
                f"and place at data/welfake/WELFake_Dataset.csv"
            )

        df = pd.read_csv(path)

        # Combine title + text for richer signal
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df["text"] = df["text"].str.strip()

        # WELFake: 0 = fake, 1 = real
        # ISOT convention: 0 = real, 1 = fake — flip to match
        df["label"] = 1 - df["label"]

        df["source_dataset"] = "welfake"

        # Drop rows with empty text
        df = df[df["text"].str.len() > 10]

        return df[["text", "label", "source_dataset"]]

    def load_combined(self):

        isot_df = self.load_isot()
        welfake_df = self.load_welfake()

        print(f"ISOT:    {len(isot_df)} samples")
        print(f"WELFake: {len(welfake_df)} samples")

        # Sample WELFake to match ISOT size — prevents domination
        welfake_sample = welfake_df.sample(
            n=min(len(isot_df), len(welfake_df)),
            random_state=42
        )

        df = pd.concat([isot_df, welfake_sample], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Combined: {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        return df[["text", "label", "source_dataset"]]

    def load_domain_testing(self):

        domain_path = self.data_dir / "Domain_testing_dataset"

        if not domain_path.exists():
            raise FileNotFoundError(f"Domain dataset not found: {domain_path}")

        files = {
            "gossipcop_fake.csv": 1,
            "gossipcop_real.csv": 0,
            "politifact_fake.csv": 1,
            "politifact_real.csv": 0,
        }

        dfs = []

        for file, label in files.items():

            file_path = domain_path / file
            df = pd.read_csv(file_path)

            if "title" not in df.columns:
                raise ValueError(f"{file} does not contain 'title' column")

            df = df[["title"]].rename(columns={"title": "text"})
            df["label"] = label
            df["source_dataset"] = "fakenewsnet"

            dfs.append(df)

        domain_df = pd.concat(dfs, ignore_index=True)

        return domain_df

    def unify_datasets(self):
        df = self.load_isot()
        df = df[["text", "label", "source_dataset"]]
        return df