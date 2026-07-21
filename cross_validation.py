import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/order_ride_features.csv")

X = df.drop(columns=["ride_requests", "request_time"])
y = df["ride_requests"]

model = RandomForestRegressor(random_state=42)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross-validation scores:", scores)
print("Average R²:", scores.mean())