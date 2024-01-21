import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

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

# Ensure that 'normalized_data' index is a datetime for proper plotting
normalized_data.index = pd.to_datetime(normalized_data.index)

# Daily Usage Chart
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
