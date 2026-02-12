import pandas as pd

df = pd.read_csv("data/raw/Matches.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Check unique values in Division column
if "Division" in df.columns:
    print("\nUnique Division values:")
    print(df["Division"].unique())
else:
    print("\nNo 'Division' column found.")
