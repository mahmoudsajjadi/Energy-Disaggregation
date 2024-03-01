from nilm_analyzer.loaders import REFIT_Loader
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import forecast

data_path = 'C:/Users/Sanic/Downloads/Processed_Data_CSV'
refit = REFIT_Loader(data_path=data_path)

house_data = refit.get_house_data(house=2)

for appliance in house_data.columns:
    if appliance.lower() in ['time', 'unix', 'aggregate']:
        continue

    appliance_data = house_data[appliance].resample('M').sum()

    model = ARIMA(appliance_data, order=(1, 1, 1))
    model_fit = model.fit()

    print(f'Forecasted energy consumption for {appliance} next month: {forecast[0]} kWh')

    plt.figure(figsize=(10, 5))
    appliance_data.plot(label=f'Historical Consumption for {appliance}')
    plt.scatter(appliance_data.index[-1], forecast, color='red', label='Forecast', zorder=5)
    plt.title(f'Energy Consumption Forecast for {appliance}')
    plt.xlabel('Month')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.show()
