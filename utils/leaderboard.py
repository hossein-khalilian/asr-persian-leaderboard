import os

import pandas as pd


def update_leaderboard(entry: dict, csv_path: str = "leaderboard.csv") -> None:
    """
    Updates the leaderboard CSV with a new entry.
    - Ensures all columns in entry and df align.
    - Updates row if a full-duplicate exists with worse WER.
    - Adds new row otherwise and re-ranks by WER (%).
    - If the file does not exist, creates a new one.
    """
    # Load or initialize leaderboard
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
    else:
        print("📁 Leaderboard file not found or empty — creating a new one.")
        df = pd.DataFrame(columns=entry.keys())

    # Ensure all entry keys exist in df columns
    for key in entry:
        if key not in df.columns:
            df[key] = None

    # Ensure all df columns exist in entry
    for col in df.columns:
        if col not in entry:
            entry[col] = None

    # Create DataFrame from entry with aligned columns
    entry_df = pd.DataFrame([entry])[df.columns.tolist()]

    print("✅ New entry — adding to leaderboard.")
    df = pd.concat([df, entry_df], ignore_index=True)

    # Recalculate rank based on WER
    df["Rank"] = df["WER (%)"].rank(method="min").astype(int)
    df = df.sort_values("WER (%)").reset_index(drop=True)

    df = df.drop_duplicates(ignore_index=True)

    # Save updated leaderboard
    df.to_csv(csv_path, index=False)
    print(f"📊 Leaderboard saved: {csv_path}")
