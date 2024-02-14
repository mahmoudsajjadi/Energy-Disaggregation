import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = 'C:/Users/Sanic/Downloads/Processed_Data_CSV'
#
house_numbers = [1, 2, 3, 4]

for house_number in house_numbers:
    house_file = f'House_{house_number}.csv'

    house_data = pd.read_csv(os.path.join(data_path, house_file), index_col='Time', parse_dates=True)

    # Aggregate the data to monthly energy consumption
    # The data is sampled every 8 seconds, hence the need to convert it to kWh for monthly aggregation
    # (Watts * seconds) / (3600 seconds/hour * 1000 Watts/kW) = kWh
    house_data['Aggregate_kWh'] = (house_data['Aggregate'] * 8) / (3600 * 1000)
    monthly_consumption = house_data['Aggregate_kWh'].resample('M').sum()

    plt.figure(figsize=(10, 5))
    monthly_consumption.plot(kind='bar')
    plt.title(f'Monthly Energy Consumption for House {house_number} (in kWh)')
    plt.xlabel('Month')
    plt.ylabel('Energy Consumption (kWh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
