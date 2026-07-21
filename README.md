# Order Ride

## Project Description

Order Ride is a machine learning project that predicts ride demand for specific locations based on historical ride request data. The project uses ride request information such as request time, latitude, longitude, pickup location, and drop-off location to forecast future ride demand. Accurate predictions help ride-hailing companies improve driver allocation, reduce passenger waiting time, and optimize fleet management.

## Data Preprocessing

The preprocessing stage includes:

- Loading the ride request dataset
- Handling missing values
- Removing duplicate records
- Standardizing column names
- Visualizing ride request patterns
- Detecting outliers
- Encoding categorical variables
- Exporting a cleaned dataset for machine learning

## Phase 2 – Clustering and Demand Prediction

This phase extends the preprocessing work by grouping nearby ride requests into geographical clusters using the K-Means clustering algorithm. Feature engineering is then performed to extract useful temporal information such as request hour, day, and weekday.

A Random Forest Regressor is trained using the engineered features to predict ride demand. The model is evaluated using regression metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R² score.