# **Data Preprocessing**
# Defined feature matrix X and target variable y.

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Import r2_score here
from xgboost import XGBRegressor
import pickle

# Load the dataset
df = pd.read_csv('SeoulBike.csv')

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Hour'] = df['Hour'].astype(int)
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Weekday'] = df['Date'].dt.weekday
df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Encoding categorical columns
le = LabelEncoder()
df['Seasons'] = le.fit_transform(df['Seasons'])
df['Holiday'] = le.fit_transform(df['Holiday'])
df['Functioning Day'] = le.fit_transform(df['Functioning Day'])

# Define features (X) and target (y)
X = df.drop(['Date', 'Rented Bike Count'], axis=1)
y = df['Rented Bike Count']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R-squared:", r2)

# Saving the trained model to a pickle file
with open('bike_demand_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Feature importance visualization
importances = model.feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar', color='salmon')
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.show()

# Alternative Model: XGBoost (for comparison)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Model Evaluation (XGBoost)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)  # Use r2_score from sklearn

print("XGBoost RMSE:", rmse_xgb)
print("XGBoost R-squared:", r2_xgb)
