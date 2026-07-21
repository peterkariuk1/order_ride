import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("data/order_ride_processed.csv")

X = df[["request_latitude", "request_longitude"]]


kmeans = KMeans(n_clusters=5, random_state=42)


df["cluster"] = kmeans.fit_predict(X)

df.to_csv("data/order_ride_clustered.csv", index=False)

plt.figure(figsize=(8,6))

plt.scatter(
    df["request_longitude"],
    df["request_latitude"],
    c=df["cluster"],
    cmap="viridis"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Ride Request Clusters")

plt.show()