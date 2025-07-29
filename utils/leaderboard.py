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


def parse_duration(dur_str):
    if pd.isna(dur_str):
        return float("inf")
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


def update_leaderboard(entry: dict, csv_path: str = "leaderboard.csv") -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
    else:
        print("📁 Leaderboard file not found or empty — creating a new one.")
        df = pd.DataFrame(columns=entry.keys())

    for key in entry:
        if key not in df.columns:
            df[key] = None
    for col in df.columns:
        if col not in entry:
            entry[col] = None

    model_name = entry["Model Name"]
    dataset = entry["Dataset Used"]
    new_wer = entry["WER (%)"]
    new_sample_size = entry.get("Sample Size", 0) or 0
    new_time_numeric = entry["Inference Time (s)"]

    mask = (df["Model Name"] == model_name) & (df["Dataset Used"] == dataset)

    if mask.any():
        idx = mask.idxmax()
        existing_wer = float(df.at[idx, "WER (%)"])
        existing_sample_size = int(df.at[idx, "Sample Size"] or 0)
        existing_time = parse_duration(df.at[idx, "Inference Time (s)"])

        should_update = (
            new_wer < existing_wer
            or new_sample_size > existing_sample_size
            or new_time_numeric < existing_time
        )

        if should_update:
            print(f"🔁 Updating existing entry for {model_name} on {dataset}.")
            entry["Inference Time (s)"] = format_duration(new_time_numeric)
            df.loc[idx] = entry

        # for col in df.columns:
        #         val = entry[col]
        #         # Handle NaNs and convert to proper type
        #         if pd.api.types.is_numeric_dtype(df[col]):
        #             if val in ("", None):
        #                 val = pd.NA
        #             else:
        #                 try:
        #                     val = float(val)
        #                 except (ValueError, TypeError):
        #                     val = pd.NA
        #         df.at[idx, col] = val

        else:
            print("⚠️ Existing entry is better or equal — skipping update.")
            return
    else:
        entry["Inference Time (s)"] = format_duration(new_time_numeric)
        entry_df = pd.DataFrame([entry])[df.columns.tolist()]
        print("✅ New entry — adding to leaderboard.")
        df = pd.concat([df, entry_df], ignore_index=True)

    df["Rank"] = df["WER (%)"].rank(method="min").astype(int)
    df = df.sort_values("WER (%)").reset_index(drop=True)
    df = df.drop_duplicates(ignore_index=True)

    df.drop("Rank", axis=1).to_csv(csv_path, index=False)
    print(f"📊 Leaderboard saved: {csv_path}")

    mask = (df["Model Name"] == model_name) & (df["Dataset Used"] == dataset)
    matched_row = df.loc[mask].iloc[0].to_dict()

    display_keys = [
        "Rank",
        "Model Name",
        "WER (%)",
        "CER (%)",
        "Inference Time (s)",
        "Dataset Used",
        "Hardware Info",
    ]
    print("\n📌 Resulting Entry Summary:")
    for key in display_keys:
        print(f"{key}: {matched_row.get(key, 'N/A')}")
