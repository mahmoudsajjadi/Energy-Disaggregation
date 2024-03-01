import pandas as pd
import gc
import matplotlib.pyplot as plt

del gc.garbage[:]

data_path = 'Datasets/PECAN_St/1minute_data_california/1minute_data_california.csv'
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
dfs_by_home_id = {home_id: df for home_id, df in dfs_by_home_id.items() if len(df) >= 50000}
# dfs_by_home_id[8574].head(1000).dropna(axis=1, how='all').mean()


# Process each dataframe to add total_consumption column and select required columns
for home_id, df in dfs_by_home_id.items():
    print(f'home_id: {home_id}')
    df['total_consumption'] = df['grid'].fillna(0) + df['solar'].fillna(0) + df['solar2'].fillna(0) + df['battery1'].fillna(0)
    dfs_by_home_id[home_id] = df[['dataid', 'localminute', 'leg1v', 'leg2v', 'total_consumption', 'refrigerator1']]
    dfs_by_home_id[home_id] = dfs_by_home_id[home_id].sort_values(by='localminute')

# Print information for each home
for home_id, df in dfs_by_home_id.items():
    print(f"Home ID: {home_id}, Number of rows: {len(df)}")

print("Total number of chunks:", total_chunks)
print("Total number of rows:", total_rows)
print("Unique dataids:", unique_homes_id)
print("Total number of unique dataids:", len(unique_homes_id))


num_min_daily = 1440

daily_consumption_by_home = {}
refrigerator_home = {}

for home_id, df in dfs_by_home_id.items():
    df[['date', 'time']] = df['localminute'].str.split(" ", expand=True)
    result = df[df['date'].map(df['date'].value_counts()) == num_min_daily].groupby('date')['total_consumption'].sum().reset_index()
    result_refrigrator = df[df['date'].map(df['date'].value_counts()) == num_min_daily].groupby('date')['refrigerator1'].sum().reset_index()
    result['total_consumption'] /= 60 * 24
    result_refrigrator['refrigerator1'] /= 60 * 24
    daily_consumption_by_home[home_id] = result
    refrigerator_home[home_id] = result_refrigrator
    
start_date = result.iloc[0]['date']
end_date = result.iloc[-1]['date']

from meteostat import Stations, Daily
# Latitude and Longitude of California
lat = 36.7783
lon = -119.4179
home_id = 8574

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

# Create a figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Counter to keep track of the number of homes plotted
homes_plotted = 0

# Iterate over the homes in the daily_consumption_by_home dictionary
for home_id, result in daily_consumption_by_home.items():
    if homes_plotted >= 4:
        break  # Stop after plotting 4 homes

    # Plot the daily consumption for the current home
    axes[homes_plotted].plot(result['date'], refrigerator_home[home_id]['refrigerator1'], label=f'Refrigerator Home {home_id}')
    # axes[homes_plotted].plot(result['date'], refrigerator_home[home_id]['refrigerator1'] * 100, label=f'Refrigerator Home {home_id}')
    axes[homes_plotted].plot(result['date'], result['total_consumption'], label=f'Total Energy Home {home_id}')
    axes[homes_plotted].set_title(f'Home {home_id}')
    axes[homes_plotted].set_xlabel('Time')
    axes[homes_plotted].set_ylabel('Daily Consumption (kWh)')
    axes[homes_plotted].legend()

    # Set x-axis tick labels to display every nth date
    n = len(result) // 10  # Adjust the value of n to display fewer or more dates
    axes[homes_plotted].set_xticks(result['date'][::n])
    axes[homes_plotted].set_xticklabels(result['date'][::n], rotation=45)  # Rotate labels for better visibility

    homes_plotted += 1

# Adjust layout and display the plot
plt.tight_layout()
plt.show()























############ training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


home_id = 8574
input_data = []
output_data = []
refrigrator = []

# Iterate over each date in the DataFrame
for date in dfs_by_home_id[home_id]['date'].unique():
    # Filter the DataFrame for the current date
    df_date = dfs_by_home_id[home_id][dfs_by_home_id[home_id]['date'] == date]
    # Check if the number of data points for the current date is equal to 60 * 24
    if len(df_date) == 60 * 24:
        # Append the data points to the input_data
        input_data.append(df_date['total_consumption'].values.reshape(60, 24))
        # Retrieve the corresponding total consumption for the next day from daily_consumption_by_home
        next_day_consumption = daily_consumption_by_home[home_id][daily_consumption_by_home[home_id]['date'] == date]['total_consumption'].values
        # Append the next day's consumption to the output_data
        output_data.append(next_day_consumption)

# Convert input and output data to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)


# Define the model architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(60 * 24, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = MLP()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train the model
num_epochs = 10
batch_size = 64
dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


losses = []  # List to store the loss values


for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Ensure targets have the same shape as the model's output
        targets = targets.view(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.train()



# Define a threshold for correctness
threshold = 0.01  # For example, consider predictions within 0.1 kWh of the ground truth as correct

# Initialize counters for correct and total predictions
correct_predictions = 0
total_predictions = 0

# Evaluate the model
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        predictions = outputs.squeeze()
        for pred, target in zip(predictions, targets):
            if abs(pred - target) <= threshold:
                correct_predictions += 1
            total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
print(f'Accuracy: {accuracy:.2f}%')


# Plot the loss values
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()












############# LSTM


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define LSTM model
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
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out

# Prepare data
home_data = {
    8574: daily_consumption_by_home[8574]['total_consumption'],
    # 3687: daily_consumption_by_home[3687]['total_consumption'],
    # 9213: daily_consumption_by_home[9213]['total_consumption'],
    # 6377: daily_consumption_by_home[6377]['total_consumption'],
    # 7062: daily_consumption_by_home[7062]['total_consumption']
}



# from meteostat import Stations, Daily
# # Latitude and Longitude of California
# lat = 36.7783
# lon = -119.4179

# # Start and end date for the data query
# # start_date = '2014-07-01'
# # end_date = '2014-08-12'

# # Get the nearest weather station
# stations = Stations()
# station = stations.nearby(lat, lon).fetch(1)
# if not station.empty:
#     station_id = station.index[0]
    
#     # Query daily weather data
#     weather_data = Daily(station_id, start=start_date, end=end_date)
#     weather_data = weather_data.fetch()
#     print(weather_data.head())
# else:
#     print('No weather station found for the given location.')
    
    

# Define sequence length
sequence_length = 7

# Define the home ID for which you want to prepare sequences
target_home_id = 8574  # Change this to the desired home ID

# Prepare input-output sequences for the target home
sequences = []
data = home_data[target_home_id]
weather_data = weather_data['tavg']
for i in range(len(data) - sequence_length):
    seq_x_power = data.iloc[i:i+sequence_length].values
    seq_x_weather = weather_data.iloc[i:i+sequence_length].values
    # seq_x = np.concatenate((seq_x, seq_x), axis=0)
    seq_x = np.vstack((seq_x_power, seq_x_weather))
    seq_y = data.iloc[i+sequence_length]
    # seq_y = refrigerator_home[target_home_id]['refrigerator1'].iloc[i+sequence_length]
    sequences.append((seq_x, seq_y))


# Define model parameters
input_size = sequence_length
hidden_size = 128
num_layers = 1

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 250
for epoch in range(num_epochs):
    if epoch % 10 == 0:
        print(epoch)
    for seq_x, seq_y in sequences:
        seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
        seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0)
        
        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()

# Make predictions
predicted_consumptions = {}
with torch.no_grad():
    for seq_x, seq_y in sequences:
        seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
        prediction = model(seq_x).item()
        predicted_consumptions[seq_y] = prediction



print("Predicted consumptions for the next day:")
for observed, prediction in predicted_consumptions.items():
    print(f"Observed consumption {observed:.2f} kWh: Predicted {prediction:.2f} kWh")




observed_consumptions = list(predicted_consumptions.keys())
predicted_consumptions = list(predicted_consumptions.values())

indices = range(len(observed_consumptions))
plt.plot(indices, observed_consumptions, label='Observed')
plt.plot(indices, predicted_consumptions, label='Predicted')
plt.xlabel('Day')
plt.ylabel('Consumption (kWh)')
plt.title('Observed vs Predicted Consumptions')

plt.legend()

plt.show()