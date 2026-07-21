import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/order_ride_features.csv")

X = df.drop(columns=["ride_requests", "request_time"])
y = df["ride_requests"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Actual vs Predicted
plt.figure(figsize=(7,5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Ride Demand")
plt.ylabel("Predicted Ride Demand")
plt.title("Actual vs Predicted Ride Demand")

plt.show()