import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/order_ride_features.csv")

y = df["ride_requests"]

X = df.drop(columns=["ride_requests", "request_time"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

print("Model trained successfully.")