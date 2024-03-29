import pandas as pd
import gc
import matplotlib.pyplot as plt

del gc.garbage[:]

data_path = 'Datasets/PECAN_St/15minute_data_newyork/15minute_data_newyork.csv'
chunk_size = 30000  # Adjust the chunk size according to your memory constraints

chunks = pd.read_csv(data_path, chunksize=chunk_size)

total_rows = 0
total_chunks = 0
unique_homes_id = set()
dfs_by_home_id = {}

# Read only the first 10 chunks
for chunk_number, chunk in enumerate(chunks, 1):
    if chunk_number > 10:
        break  # Stop after reading 10 chunks

    total_chunks += 1
    print(f"Reading chunk {chunk_number}")
    total_rows += len(chunk)

    for home_id in chunk['dataid'].unique():
        home_data = chunk[chunk['dataid'] == home_id]

        if home_id in dfs_by_home_id:
            dfs_by_home_id[home_id] = pd.concat([dfs_by_home_id[home_id], home_data])
        else:
            dfs_by_home_id[home_id] = home_data

# Filter out homes with less than 500,000 rows
dfs_by_home_id = {home_id: df for home_id, df in dfs_by_home_id.items() if len(df) >= 10000}


home_ids_to_find = [10202, 8577, 2126, 3000, 9973, 5679, 5587, 27, 1222, 3517, 6178, 518, 5058, 6823] # has car in NY

keys = dfs_by_home_id.keys()

found = False
found_id = None
for key in keys:
    if key in home_ids_to_find:
        found = True
        found_id = key
        break

if found:
    print("At least one of the home IDs was found:", found_id)
else:
    print("None of the home IDs were found.")
    
    unique_home_ids = set()
    # Read chunks until data for one of the home IDs is found
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        for home_id in chunk['dataid'].unique():
            if home_id not in unique_home_ids:
                print("Found data for new home ID:", home_id)
                unique_home_ids.add(home_id)
        for home_id in home_ids_to_find:
            if home_id in unique_home_ids:
                
            # home_data = chunk[chunk['dataid'] == home_id]
            # if not home_data.empty:
                print("Found data for home ID:", home_id)
                print(home_data)
                break
        else:
            continue  # Continue to the next chunk if no data is found for any of the home IDs
        break  # Break the loop if data is found for one of the home IDs
    
    




# Process each dataframe to add total_consumption column and select required columns
for home_id, df in dfs_by_home_id.items():
    print(f'home_id: {home_id}')
    df['total_consumption'] = df['grid'].fillna(0) + df['solar'].fillna(0) + df['solar2'].fillna(0) + df['battery1'].fillna(0)
    dfs_by_home_id[home_id] = df[['dataid', 'local_15min', 'leg1v', 'leg2v', 'total_consumption', 'refrigerator1', 'car1']]
    dfs_by_home_id[home_id] = dfs_by_home_id[home_id].sort_values(by='local_15min')

# Print information for each home
for home_id, df in dfs_by_home_id.items():
    print(f"Home ID: {home_id}, Number of rows: {len(df)}")

print("Total number of chunks:", total_chunks)
print("Total number of rows:", total_rows)
print("Unique dataids:", unique_homes_id)
print("Total number of unique dataids:", len(unique_homes_id))


num_15min_daily = 96
num_measure_hour = 4

daily_consumption_by_home = {}
refrigerator_home = {}

varible_to_pridict = 'total_consumption'
# varible_to_pridict = 'car1'

for home_id, df in dfs_by_home_id.items():
    df[['date', 'time']] = df['local_15min'].str.split(" ", expand=True)
    result = df[df['date'].map(df['date'].value_counts()) == num_15min_daily].groupby('date')[varible_to_pridict].sum().reset_index()
    result_refrigrator = df[df['date'].map(df['date'].value_counts()) == num_measure_hour].groupby('date')['refrigerator1'].sum().reset_index()
    result[varible_to_pridict] /= num_measure_hour
    result_refrigrator['refrigerator1'] /= num_measure_hour
    daily_consumption_by_home[home_id] = result
    refrigerator_home[home_id] = result_refrigrator
    
start_date = result.iloc[0]['date']
end_date = result.iloc[-1]['date']

from meteostat import Stations, Daily
# Latitude and Longitude of California
lat = 36.7783
lon = -119.4179
home_id = 5679

# Start and end date for the data query
# start_date = '2014-07-01'
# end_date = '2014-08-12'

# Get the nearest weather station
stations = Stations()
station = stations.nearby(lat, lon).fetch(1)
if not station.empty:
    station_id = station.index[0]
    
    # Query daily weather data
    weather_data = Daily(station_id, start=start_date, end=end_date)
    weather_data = weather_data.fetch()
    print(weather_data.head())
else:
    print('No weather station found for the given location.')


        
# Convert the 'date' column to datetime objects
daily_consumption_by_home[home_id]['date'] = pd.to_datetime(daily_consumption_by_home[home_id]['date'])

# Iterate through each date in the weather data index
for date in weather_data.index:
    # Extract the date part without considering the time
    date_without_time = date.date()

    # Check if there is a corresponding entry in the consumption data
    if date_without_time not in daily_consumption_by_home[home_id]['date'].dt.date.values:
        # Remove the row from the weather data if there is no entry for that date
        weather_data = weather_data.drop(index=date)





############ training

import torch
import torch.nn as nn


# home_id = 5679
# input_data = []
# output_data = []
# refrigrator = []

# # Iterate over each date in the DataFrame
# for date in dfs_by_home_id[home_id]['date'].unique():
#     # Filter the DataFrame for the current date
#     df_date = dfs_by_home_id[home_id][dfs_by_home_id[home_id]['date'] == date]
#     # Check if the number of data points for the current date is equal to 60 * 24
#     if len(df_date) == num_15min_daily:
#         input_data.append(df_date[varible_to_pridict].values.reshape(4, 24))
#         next_day_consumption = daily_consumption_by_home[home_id][daily_consumption_by_home[home_id]['date'] == date][varible_to_pridict].values
#         output_data.append(next_day_consumption)

# # Convert input and output data to PyTorch tensors
# input_tensor = torch.tensor(input_data, dtype=torch.float32)
# output_tensor = torch.tensor(output_data, dtype=torch.float32)




# Define function to collect data for a given home ID
def collect_data_for_home(home_id):
    input_data = []
    output_data = []

    # Iterate over each date in the DataFrame
    for date in dfs_by_home_id[home_id]['date'].unique():
        # Filter the DataFrame for the current date
        df_date = dfs_by_home_id[home_id][dfs_by_home_id[home_id]['date'] == date]
        # Check if the number of data points for the current date is equal to 60 * 24
        if len(df_date) == num_15min_daily:
            input_data.append(df_date[varible_to_pridict].values.reshape(4, 24))
            next_day_consumption = daily_consumption_by_home[home_id][daily_consumption_by_home[home_id]['date'] == date][varible_to_pridict].values
            output_data.append(next_day_consumption)

    # Convert input and output data to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)

    return input_tensor, output_tensor

# Collect data for home ID 5679
input_tensor_5679, output_tensor_5679 = collect_data_for_home(5679)

# Collect data for home ID 4550
input_tensor_4550, output_tensor_4550 = collect_data_for_home(4550)

# Concatenate input and output tensors for both home IDs
input_combined = torch.cat((input_tensor_5679, input_tensor_4550), dim=0)
output_combined = torch.cat((output_tensor_5679, output_tensor_4550), dim=0)






############# LSTM


import torch
import numpy as np
import pandas as pd


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)


# Prepare data
home_data = {
    5679: daily_consumption_by_home[5679][varible_to_pridict],
    4550: daily_consumption_by_home[4550][varible_to_pridict],
    3700: daily_consumption_by_home[3700][varible_to_pridict],
    2318: daily_consumption_by_home[2318][varible_to_pridict],
    # 5058: daily_consumption_by_home[5058][varible_to_pridict]
}



  
    

# Define sequence length
sequence_length = 14

# Define the home ID for which you want to prepare sequences
target_home_id = 5679  # Change this to the desired home ID

# Prepare input-output sequences for the target home
sequences_5679 = []
sequences_4550 = []

# data = home_data[target_home_id]
weather_data = weather_data['tavg']
# Initialize lists to store sequences for each home
sequences_5679 = []
sequences_4550 = []
sequences_3700 = []
sequences_2318 = []

# Prepare input-output sequences for each home
for target_home_id, data in home_data.items():
    sequences = []
    # weather_data = pd.read_csv('weather_data.csv')  # Assuming weather data is stored in a CSV file
    
    # # Ensure that 'tavg' column is present in weather_data DataFrame
    # if 'tavg' not in weather_data.columns:
    #     raise KeyError("Column 'tavg' not found in weather_data DataFrame.")
    
    for i in range(len(data) - sequence_length):
        seq_x_power = data.iloc[i:i+sequence_length].values
        seq_x_weather = weather_data.iloc[i:i+sequence_length].values
        # seq_x_weather = weather_data
        seq_x = np.vstack((seq_x_power, seq_x_weather))
        seq_y = data.iloc[i+sequence_length]
        if target_home_id == 5679:
            sequences_5679.append((seq_x, seq_y))
        elif target_home_id == 4550:
            sequences_4550.append((seq_x, seq_y))
        elif target_home_id == 3700:
            sequences_3700.append((seq_x, seq_y))
        elif target_home_id == 2318:
            sequences_2318.append((seq_x, seq_y))

# Define model parameters
input_size = sequence_length
hidden_size = 128
num_layers = 1

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for each home
num_epochs = 100


predicted_consumptions = {}

# Iterate over each home
# for target_home_id, sequences in zip(home_data.keys(), [sequences_5679, sequences_4550, sequences_3700, sequences_2318]):
#     # Split data into training and testing sets
#     # sequences_train, sequences_test = train_test_split(sequences_train, test_size=0.2, random_state=42)

#     # Initialize model, loss function, and optimizer
#     model = LSTMModel(input_size, hidden_size, num_layers)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     for epoch in range(num_epochs):
#         if epoch % 10 == 0:
#             print(f"Home ID: {target_home_id}, Epoch: {epoch}")
#         for seq_x, seq_y in sequences:
#             seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
#             seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0)

#             optimizer.zero_grad()
#             output = model(seq_x)
#             loss = criterion(output, seq_y)
#             loss.backward()
#             optimizer.step()

#     # Make predictions for the trained model
#     predicted_consumptions[target_home_id] = {}
#     with torch.no_grad():
#         for seq_x, seq_y in sequences:
#             seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
#             prediction = model(seq_x).item()
#             predicted_consumptions[target_home_id][seq_y] = prediction

#     # Print test accuracy
#     # print(f"Test Accuracy for Home ID {target_home_id}: {test_losses[-1]}")
    

from torch.utils.data import ConcatDataset, DataLoader

# Combine all sequences
all_sequences = ConcatDataset([sequences_5679, sequences_4550, sequences_3700, sequences_2318])



# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")
    # Initialize DataLoader with shuffled data
    data_loader = DataLoader(all_sequences, batch_size=10, shuffle=True)
    for seq_x, seq_y in data_loader:
        # # seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).squeeze(0)
        # seq_x = torch.tensor(seq_x, dtype=torch.float32)
        # # seq_x = seq_x.squeeze(0)
        # seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0)
        
        seq_x = torch.tensor(seq_x, dtype=torch.float32).clone().detach()
        seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0).clone().detach()
        # seq_x = seq_x.clone().detach()
        # seq_x = torch.tensor(seq_x, dtype=torch.float32)
        # seq_y = seq_y.clone().detach()
        


        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()

# Make predictions for each home
predicted_consumptions = {}
for target_home_id, sequences in zip(home_data.keys(), [sequences_5679, sequences_4550, sequences_3700, sequences_2318]):
    predicted_consumptions[target_home_id] = {}
    with torch.no_grad():
        for seq_x, seq_y in sequences:
            seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
            prediction = model(seq_x).item()
            predicted_consumptions[target_home_id][seq_y] = prediction

# Plotting for each home
plt.figure(figsize=(22, 8))

for idx, (target_home_id, sequences) in enumerate(predicted_consumptions.items(), start=1):
    observed_consumptions = list(sequences.keys())
    predicted_consumptions = list(sequences.values())
    
    plt.subplot(2, 2, idx)  # Create subplots
    plt.plot(np.array(observed_consumptions) / 1, label='Observed')
    plt.plot(np.array(predicted_consumptions) / 1, label='Predicted')
    plt.xlabel('Day')
    plt.ylabel('Consumption (kWh)')
    plt.title(f'Sample {idx + 1}')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(0, 50)  # Set x-axis limits

# Adjust layout and display plot
plt.tight_layout()
plt.show()