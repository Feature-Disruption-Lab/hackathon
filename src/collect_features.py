import pandas as pd

from constants import DATA_DIR


def load_prompts(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

file_path = DATA_DIR / "example_data.csv"
prompts_df = load_prompts(file_path)

# Print first few rows
print(prompts_df.head())

# Get unique labels
print("\nUnique labels:", prompts_df['label'].unique())