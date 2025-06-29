import os
import pandas as pd


def update_leaderboard(entry: dict, csv_path: str = "leaderboard.csv") -> None:
    """
    Updates the leaderboard CSV with a new entry.
    - If a model+dataset combo already exists and the new WER is better, it updates the row.
    - If it's worse or equal, it skips the update.
    - Otherwise, it adds a new row and re-ranks by WER (%).
    """
    # Load or initialize leaderboard
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame([entry]).iloc[0:0]  # preserve types, avoid warning

    # Check for duplicate model+dataset combo
    mask = (df["Model Name"] == entry["Model Name"]) & (
        df["Dataset Used"] == entry["Dataset Used"]
    )

    if mask.any():
        existing_wer = df.loc[mask, "WER (%)"].values[0]
        if entry["WER (%)"] < existing_wer:
            print(
                f"🔁 Found duplicate with worse WER ({existing_wer}%) — updating to {entry['WER (%)']}%."
            )
            for key, value in entry.items():
                if key != "Rank":
                    df.loc[mask, key] = value
        else:
            print(
                f"⚠️ Duplicate exists with better or equal WER ({existing_wer}%) — skipping update."
            )
    else:
        print("✅ New entry — adding to leaderboard.")
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    # Recalculate rank based on WER
    df["Rank"] = df["WER (%)"].rank(method="min").astype(int)
    df = df.sort_values("WER (%)").reset_index(drop=True)

    # Save updated leaderboard
    df.to_csv(csv_path, index=False)
    print(f"📊 Leaderboard saved: {csv_path}")
