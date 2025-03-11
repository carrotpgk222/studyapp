import pandas as pd
import random

# Load study data
data = pd.read_csv("../train/study.csv")

# Step 1: Select subjects with high satisfaction (4-5) & medium difficulty (2-4)
preferred = data[(data["On a scale of 1 to 5 how satisfied your study session was"] >= 4) &
                 (data["On a scale of 1 (easy) to 5 (very challenging), how difficult do you find each subject?"].between(2, 4))]

if not preferred.empty:
    selected_subject = preferred.sample(n=1)["what subject are you doing today?"].values[0]
    print(f"📌 Recommended Subject: {selected_subject}")
else:
    # Step 2: Check for Medium Difficulty (2-4) & Low Satisfaction (1-2) → 30% probability
    secondary = data[(data["On a scale of 1 to 5 how satisfied your study session was"] <= 2) &
                     (data["On a scale of 1 (easy) to 5 (very challenging), how difficult do you find each subject?"].between(2, 4))]

    if not secondary.empty and random.random() < 0.3:
        selected_subject = secondary.sample(n=1)["what subject are you doing today?"].values[0]
        print(f"🛠 Encouraging study of: {selected_subject}")
    else:
        # Step 3: Medium satisfaction (≈3) & medium difficulty (≈3)
        fallback = data[(data["On a scale of 1 to 5 how satisfied your study session was"] == 3) &
                        (data["On a scale of 1 (easy) to 5 (very challenging), how difficult do you find each subject?"] == 3)]

        if not fallback.empty:
            selected_subject = fallback.sample(n=1)["what subject are you doing today?"].values[0]
            print(f"⚖ Balanced option: {selected_subject}")
        else:
            print("❌ No suitable subjects found.")
