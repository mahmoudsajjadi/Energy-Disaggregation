import pandas as pd
import sns as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
#watashi wa americajin desu
file_path = 'cleaned_normalized_energy_data.csv'
normalized_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

weekly_totals = normalized_data.resample('W').sum()

train_weekly, test_weekly = train_test_split(weekly_totals, test_size=0.2, shuffle=False)

weekly_feature_column = weekly_totals.columns[0]

arima_weekly_model = ARIMA(train_weekly[weekly_feature_column], order=(1,1,1))
arima_weekly_model_fit = arima_weekly_model.fit()
arima_weekly_forecast = arima_weekly_model_fit.forecast(steps=len(test_weekly))

rf_weekly_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_weekly_model.fit(train_weekly[[weekly_feature_column]], train_weekly[weekly_feature_column])
rf_weekly_predictions = rf_weekly_model.predict(test_weekly[[weekly_feature_column]])

arima_weekly_mae = mean_absolute_error(test_weekly[weekly_feature_column], arima_weekly_forecast)
arima_weekly_rmse = sqrt(mean_squared_error(test_weekly[weekly_feature_column], arima_weekly_forecast))
rf_weekly_mae = mean_absolute_error(test_weekly[weekly_feature_column], rf_weekly_predictions)
rf_weekly_rmse = sqrt(mean_squared_error(test_weekly[weekly_feature_column], rf_weekly_predictions))

print("ARIMA Weekly MAE:", arima_weekly_mae)
print("ARIMA Weekly RMSE:", arima_weekly_rmse)
print("Random Forest Weekly MAE:", rf_weekly_mae)
print("Random Forest Weekly RMSE:", rf_weekly_rmse)

normalized_data.index = pd.to_datetime(normalized_data.index)

#plotss
plt.figure(figsize=(12, 6))
plt.plot(normalized_data[weekly_feature_column], label='Daily Usage')
plt.title('Daily Energy Usage')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(weekly_totals[weekly_feature_column], label='Weekly Total Usage')
plt.title('Weekly Energy Usage')
plt.xlabel('Week')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

monthly_totals = normalized_data.resample('M').sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_totals[weekly_feature_column], label='Monthly Total Usage')
plt.title('Monthly Energy Usage')
plt.xlabel('Month')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

normalized_data['DayOfWeek'] = normalized_data.index.dayofweek
normalized_data['Month'] = normalized_data.index.month

for lag in range(1, 4):
    normalized_data[f'Lag_{lag}'] = normalized_data[weekly_feature_column].shift(lag)
normalized_data.dropna(inplace=True)

normalized_data['Rolling_Mean'] = normalized_data[weekly_feature_column].rolling(window=3).mean()
normalized_data['Rolling_Std'] = normalized_data[weekly_feature_column].rolling(window=3).std()
normalized_data.dropna(inplace=True)

print("Descriptive Statistics:")
print(normalized_data.describe())

print("\nKey Insights:")

plt.figure(figsize=(12, 6))
plt.plot(test_weekly[weekly_feature_column], label='Actual Usage')
plt.plot(test_weekly.index, arima_weekly_forecast, label='ARIMA Predicted Usage', alpha=0.7)
plt.title('Weekly Energy Usage: Actual vs ARIMA Predictions')
plt.xlabel('Week')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_weekly[weekly_feature_column], label='Actual Usage')
plt.plot(test_weekly.index, rf_weekly_predictions, label='Random Forest Predicted Usage', alpha=0.7)
plt.title('Weekly Energy Usage: Actual vs Random Forest Predictions')
plt.xlabel('Week')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

arima_residuals = test_weekly[weekly_feature_column] - arima_weekly_forecast

plt.figure(figsize=(12, 6))
plt.plot(arima_residuals)
plt.title('ARIMA Model Residuals')
plt.xlabel('Week')
plt.ylabel('Residuals')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(normalized_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()