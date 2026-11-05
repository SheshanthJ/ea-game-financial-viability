"""
Run this FIRST on the real CSVs:
    python inspect_data.py steam.csv games.csv
It prints headers, a sample row, and checks the things ea_pipeline.py depends on.
"""
import sys
import pandas as pd

old_path, new_path = sys.argv[1], sys.argv[2]
old = pd.read_csv(old_path)
new = pd.read_csv(new_path, on_bad_lines="skip")

for label, df in [("OLD (2019-style snapshot)", old), ("NEW (recent snapshot)", new)]:
    print("=" * 70)
    print(label, "-> rows:", len(df), "| cols:", len(df.columns))
    print("COLUMNS:", list(df.columns))
    print("INDEX sample (should be real app ids for NEW):", list(df.index[:5]))
    print("\nFIRST ROW:")
    print(df.iloc[0].to_string()[:1500])

    # find whichever column holds genres and count Early Access rows
    for c in df.columns:
        if "genre" in c.lower():
            n = df[c].fillna("").astype(str).str.contains("Early Access", case=False).sum()
            print(f"\n'{c}' column: {n} rows contain 'Early Access'")

print("=" * 70)
print("CHECK BY HAND:")
print(" 1. Does OLD have: id, release date, developer, genres, tags, owners, price, ratings, playtime, achievements?")
print(" 2. Does NEW have an id column + genres column, and do the values look right (not shifted)?")
print(" 3. Are both id columns the same kind of number (so they can be joined)?")
print(" 4. Early Access counts above should be in the hundreds/thousands, not 0.")
print(" 5. Note the scrape/update date of each file from its Kaggle page -> use for --old-date / --new-date")
