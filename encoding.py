gimport pandas as pd

df = pd.read_csv("data/order_ride_cleaned.csv")
categorical = ["pickup_location", "drop_location"]

existing = [c for c in categorical if c in df.columns]

df = pd.get_dummies(df, columns=existing)

df.to_csv("data/order_ride_processed.csv", index=False)

print(df.head())