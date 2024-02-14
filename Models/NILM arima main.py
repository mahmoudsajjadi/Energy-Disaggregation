from nilm_analyzer.loaders import REFIT_Loader
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

data_path = 'C:/Users/Sanic/Downloads/Processed_Data_CSV'
refit = REFIT_Loader(data_path=data_path)

house_data = refit.get_house_data(house=2)

house_data['Aggregate_kWh'] = (house_data['aggregate'] * 8) / (3600 * 1000)
monthly_consumption = house_data['Aggregate_kWh'].resample('M').sum()

split_point = int(len(monthly_consumption) * 0.8)  # 80% for train, 20% for test
train, test = monthly_consumption[0:split_point], monthly_consumption[split_point:]

# ARIMA Model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

forecast_values = model_fit.forecast(steps=len(test))

forecast = model_fit.forecast(steps=1)


plt.figure(figsize=(10, 5))
train.plot(label='Training Set')
test.plot(label='Test Set', color='orange', linewidth=2)
forecast_values.plot(label='Forecast', style='--', color='red', linewidth=2)
plt.title('Monthly Energy Consumption Forecast')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.xlim(right=house_data.index[-1])
plt.legend()
plt.show()

mse = mean_squared_error(test, forecast_values)

rmse = sqrt(mse)

print(f'The RMSE of the forecast is: {rmse}')
