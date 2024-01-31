import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.patches import Patch
import torch
import torch.optim as optim
import torch.nn as nn
from Models.neural_network import Net


# Specify the test house number
test_house_number = 6

# Load REDD dataset
dataset_folder = './Datasets/REDD/'
dataset_files = natsorted([f for f in os.listdir(dataset_folder) if f.endswith('.csv')])
redd_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files])

# Function to load house data
def load_house_data(file):
    return pd.read_csv(os.path.join(dataset_folder, file))

# Load house data into a dictionary
house_data_dict = {file.split('_')[1]: load_house_data(file) for file in dataset_files}

# Extract common appliances
common_appliances = set(house_data_dict[dataset_files[0].split('_')[1]].columns[1:])
all_appliances = set(house_data_dict[dataset_files[0].split('_')[1]].columns[1:])
for file in dataset_files[1:]:
    common_appliances &= set(load_house_data(file).columns[1:])
    all_appliances |= set(load_house_data(file).columns[1:])

# Define colors for appliances
appliance_colors = {
    'CE appliance': 'yellow',
    'dish washer': 'green',
    'electric furnace': 'red',
    'electric space heater': 'purple',
    'electric stove': 'orange',
    'fridge': 'cyan',
    'main': 'blue',
    'microwave': 'magenta',
    'washer dryer': 'lime',
    'waste disposal unit': 'brown'
}

# Define subplot grid
num_houses = len(house_data_dict)
num_rows = num_houses // 3 + (1 if num_houses % 3 != 0 else 0)
num_cols = min(num_houses, 3)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Flatten axes if necessary
if num_houses == 1:
    axes = [axes]

# Define legend labels and handles
legend_labels = list(appliance_colors.keys())
legend_handles = [Patch(facecolor=color, edgecolor='black', label=label) for label, color in appliance_colors.items()]

# Iterate through house data
for idx, (house_name, house_data) in enumerate(house_data_dict.items()):
    existing_appliances = set(house_data.columns[1:])
    missing_appliances = appliance_colors.keys() - existing_appliances

    # Add missing appliances with zeros if needed
    if missing_appliances:
        house_data = house_data.reindex(columns=appliance_colors.keys(), fill_value=0)

    # Calculate means
    appliance_means = house_data[list(appliance_colors.keys())].mean().sort_values(ascending=False)
    main_mean = house_data['main'].mean()

    # Visualize appliance means with bar chart
    row = idx // num_cols
    col = idx % num_cols
    ax = axes[row][col]

    # Assign colors to bars
    colors = [appliance_colors.get(appliance, "gray") for appliance in appliance_means.index]

    ax.bar(appliance_means.index, appliance_means.values, color=colors)
    ax.set_ylabel("Mean Power Consumption")
    ax.set_title(f"Mean Power Consumption of Appliances in House {house_name}")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis labels

# Ensure tight layout for the first plot
plt.tight_layout()

# Display custom legend outside the subplots for the first plot
plt.legend(handles=legend_handles, labels=legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1), title="Appliances", fancybox=True, shadow=True, ncol=1)

# Create a new figure and axes for the second plot
fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Flatten axes if necessary
if num_houses == 1:
    axes2 = [axes2]

# Iterate through house data for common appliances
for idx, (house_name, house_data) in enumerate(house_data_dict.items()):
    existing_appliances = set(house_data.columns[1:])
    common_existing_appliances = existing_appliances.intersection(common_appliances)

    # If there are common appliances present in this house's data
    if common_existing_appliances:
        # Convert set to list for indexing
        common_existing_appliances_list = list(common_existing_appliances)

        # Filter data to include only common appliances
        house_data_common = house_data[common_existing_appliances_list]

        # Calculate means
        appliance_means = house_data_common.mean().sort_values(ascending=False)
        main_mean = house_data['main'].mean()  # Still calculate main mean for information

        # Visualize appliance means with bar chart for common appliances
        row = idx // num_cols
        col = idx % num_cols
        ax = axes2[row][col]

        # Assign colors to bars using the original color dictionary
        colors = [appliance_colors.get(appliance, "gray") for appliance in appliance_means.index]

        ax.bar(appliance_means.index, appliance_means.values, color=colors)
        ax.set_ylabel("Mean Power Consumption")
        ax.set_title(f"Mean Power Consumption of Common Appliances in House {house_name}")
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis labels


# Ensure tight layout for the second plot
plt.tight_layout()

# Display custom legend outside the subplots for the second plot
plt.legend(handles=legend_handles, labels=legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1), title="Common Appliances", fancybox=True, shadow=True, ncol=1)

plt.show()




############### Model training

# Define legend labels and handles
legend_labels = list(appliance_colors.keys())
legend_handles = [Patch(facecolor=color, edgecolor='black', label=label) for label, color in appliance_colors.items()]

# Create the neural network instance
input_size = len(all_appliances)  # Input size is the number of all appliances
hidden_size = 100
output_size = len(common_appliances)  # Output size is the number of common appliances
net = Net(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Function to preprocess data for neural network
def preprocess_data(data):
    # Assuming data is a DataFrame with appliance means
    return torch.tensor(data.values.astype(np.float32))

# Train the neural network
for house_name, house_data in house_data_dict.items():
    # Preprocess input and output data
    input_data = preprocess_data(house_data[common_appliances])
    output_data = preprocess_data(house_data[common_appliances])

    # Training loop
    optimizer.zero_grad()
    outputs = net(input_data.unsqueeze(0))  # Add batch dimension
    loss = criterion(outputs, output_data.unsqueeze(0))  # Add batch dimension
    loss.backward()
    optimizer.step()

# Test the neural network
test_house_data = house_data_dict[f'house{test_house_number}']
input_data_test = preprocess_data(test_house_data['main'])
output_data_pred = net(input_data_test.unsqueeze(0))  # Add batch dimension

# Convert output tensor to DataFrame with appliance means
output_df_pred = pd.DataFrame(output_data_pred.squeeze().detach().numpy(), columns=common_appliances)
print("Predicted Appliance Means:")
print(output_df_pred)