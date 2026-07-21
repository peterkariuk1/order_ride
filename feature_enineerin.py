import pandas as pd

df = pd.read_csv("data/order_ride_clustered.csv")

df["request_time"] = pd.to_datetime(df["request_time"])

df["hour"] = df["request_time"].dt.hour
df["day"] = df["request_time"].dt.day
df["weekday"] = df["request_time"].dt.day_name()

df = pd.get_dummies(df, columns=["weekday"])

df.to_csv("data/order_ride_features.csv", index=False)

print(df.head())