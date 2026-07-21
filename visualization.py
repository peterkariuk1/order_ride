import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/order_ride_cleaned.csv")

plt.figure(figsize=(8,5))

sns.histplot(df["request_time"], bins=24)

plt.title("Ride Requests by Hour")
plt.xlabel("Request Time")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="pickup_location",
    order=df["pickup_location"].value_counts().index
)

plt.xticks(rotation=45)
plt.title("Pickup Locations")
plt.show()


plt.figure(figsize=(8,5))

sns.boxplot(x=df["request_latitude"])

plt.title("Latitude Outliers")
plt.show()

plt.figure(figsize=(6,5))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Matrix")
plt.show()