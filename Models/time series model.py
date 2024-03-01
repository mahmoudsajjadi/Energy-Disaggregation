
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

file_path = 'C:\\Users\\Sanic\\OneDrive\\Documents\\Github\\Energy-Disaggregation\\Datasets\\REDD\\redd_house1_0.csv'

data = pd.read_csv(file_path)

# Assuming each row represents a fixed time interval, aggregate data weekly
data['week'] = data.index // 168  # Assuming hourly data
weekly_data = data.groupby('week').sum()

# Create lag features for the 'main' consumption column
for i in range(1, 4):  # Create lags up to 3 weeks back
    weekly_data[f'lag_{i}'] = weekly_data['main'].shift(i)

weekly_data.dropna(inplace=True)

X = weekly_data[[f'lag_{i}' for i in range(1, 4)]]
y = weekly_data['main']

test_size = int(len(X) * 0.2)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model_gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_gbm.fit(X_train, y_train)

y_pred_gbm = model_gbm.predict(X_test)
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
print(f"Mean Absolute Error: {mae_gbm}")

#
feature_importances_gbm = model_gbm.feature_importances_
plt.figure(figsize=(10, 6))
plt.title("Feature Importances in Gradient Boosting Model")
plt.bar(range(X_train.shape[1]), feature_importances_gbm, align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Consumption', alpha=0.7)
plt.plot(y_pred_gbm, label='Predicted Consumption', alpha=0.7)
plt.title('Predicted vs Actual Energy Consumption')
plt.xlabel('Time (weeks)')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
