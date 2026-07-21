import pandas as pd

# Load dataset
df = pd.read_csv("data/order_ride.csv")


print(df.head())


print(df.info())

print(df.isnull().sum())


df.drop_duplicates(inplace=True)

numeric_cols = ["request_latitude", "request_longitude"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())


categorical_cols = ["pickup_location", "drop_location"]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df.columns = df.columns.str.lower().str.replace(" ", "_")

# Save cleaned dataset
df.to_csv("data/order_ride_cleaned.csv", index=False)

print("Data cleaning completed.")