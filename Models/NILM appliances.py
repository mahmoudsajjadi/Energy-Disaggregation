from nilm_analyzer.loaders import REFIT_Loader
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = 'C:/Users/Sanic/Downloads/Processed_Data_CSV'
refit = REFIT_Loader(data_path=data_path)

# List of households
house_numbers = [20]

exclude_columns = ['aggregate']

for house_number in house_numbers:
    house_data = refit.get_house_data(house=house_number)

    appliance_columns = [col for col in house_data.columns if col not in exclude_columns]

    for appliance in appliance_columns:
        # Convert appliance energy usage from Watts to kWh
        house_data[f'{appliance}_kWh'] = (house_data[appliance] * 8) / (3600 * 1000)

        # Calculate monthly consumption for the appliance
        monthly_consumption = house_data[f'{appliance}_kWh'].resample('M').sum()

        plt.figure(figsize=(10, 5))
        monthly_consumption.plot(kind='bar')
        plt.title(f'Monthly Energy Consumption for {appliance} in House {house_number} (in kWh)')
        plt.xlabel('Month')
        plt.ylabel('Energy Consumption (kWh)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
