import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/order_ride_features.csv")

# Features and target
X = df.drop(columns=["ride_requests", "request_time"])
y = df["ride_requests"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n{name}")

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
    print("R²:", r2_score(y_test, predictions))