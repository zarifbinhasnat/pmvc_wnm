import re
import pandas as pd


def preprocess_text(text: str) -> str:
    """Clean a raw Banglish string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset and apply preprocessing."""
    df = pd.read_csv(csv_path)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # rename to standard names if needed
    col_map = {}
    for col in df.columns:
        if 'banglish' in col or 'roman' in col:
            col_map[col] = 'banglish'
        elif 'bengali' in col or 'bangla' in col:
            col_map[col] = 'bengali'
        elif 'label' in col or 'emotion' in col or 'sentiment' in col:
            col_map[col] = 'label'
    df.rename(columns=col_map, inplace=True)

    # drop nulls
    df.dropna(subset=['banglish', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # clean text
    df['clean'] = df['banglish'].apply(preprocess_text)
    df = df[df['clean'].str.len() > 2].reset_index(drop=True)

    # Ditch the native Bengali script column: the model operates ONLY on the
    # romanized Banglish text. Datasets like b-and-b-80k ship a parallel
    # 'Bengali' column; we drop it so it can never leak into features and so
    # memory isn't wasted carrying ~80k rows of unused text.
    dropped = [c for c in ('bengali',) if c in df.columns]
    df = df[[c for c in ('banglish', 'clean', 'label') if c in df.columns]]
    if dropped:
        print(f"Dropped native-script column(s): {dropped} (model uses Banglish only)")

    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df
