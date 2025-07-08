import os

import pandas as pd


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def update_leaderboard(entry: dict, csv_path: str = "leaderboard.csv") -> None:
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

    # Save numeric inference time for comparison
    new_time_numeric = entry["Inference Time (s)"]

    # Columns to check for equality to decide update
    key_cols = ["Model Name", "WER (%)", "Dataset Used", "Sample Size", "Hardware Info"]

    # Create mask where all key columns equal
    mask = pd.Series([True] * len(df))
    for col in key_cols:
        mask &= df[col] == entry[col]

    def parse_duration(dur_str):
        # Parse string like '1h 2m 30s' to seconds
        if pd.isna(dur_str):
            return float("inf")  # Treat missing times as very large
        parts = dur_str.split()
        total_sec = 0
        for part in parts:
            if part.endswith("h"):
                total_sec += int(part[:-1]) * 3600
            elif part.endswith("m"):
                total_sec += int(part[:-1]) * 60
            elif part.endswith("s"):
                total_sec += int(part[:-1])
        return total_sec

    if mask.any():
        # There is at least one matching row
        idx = mask.idxmax()  # Get first matching row index

        existing_time = parse_duration(df.at[idx, "Inference Time (s)"])

        if new_time_numeric < existing_time:
            print(
                f"🔁 Updating inference time from {df.at[idx, 'Inference Time (s)']} to {format_duration(new_time_numeric)} for better time."
            )
            df.at[idx, "Inference Time (s)"] = format_duration(new_time_numeric)

            # Update other columns (except Rank)
            for col in df.columns:
                if col != "Rank":
                    df.at[idx, col] = entry[col]
        else:
            print(
                "⚠️ Existing entry has equal or better inference time — skipping update."
            )
            return
    else:
        # No matching row — add new
        entry["Inference Time (s)"] = format_duration(new_time_numeric)
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
