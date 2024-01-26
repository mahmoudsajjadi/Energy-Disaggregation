import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.patches import Patch

# Specify the test house number
test_house_number = 6

# Load REDD dataset
# dataset_folder = '/home/seyedmahmouds/Projects/Energy Disaggregation/Energy-Disaggregation/Datasets/REDD/'
dataset_folder = './Datasets/REDD/'
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
dataset_files = natsorted(dataset_files)
redd_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files])

# Iterate over the files and create separate DataFrames for each house
house_data_dict = {}
for file in dataset_files:
    house_name = file.split('_')[1]  # Extract house name from file name
    house_data_dict[house_name] = pd.read_csv(os.path.join(dataset_folder, file))

# Iterate over the remaining files and update common appliances
common_appliances = set(pd.read_csv(os.path.join(dataset_folder, dataset_files[0]), nrows=1).columns[1:])
all_appliances = set(pd.read_csv(os.path.join(dataset_folder, dataset_files[0]), nrows=1).columns[1:])
for file in dataset_files[1:]:
    file_path = os.path.join(dataset_folder, file)
    current_columns = set(pd.read_csv(file_path, nrows=1).columns[1:])
    common_appliances = common_appliances.intersection(current_columns)
    all_appliances = all_appliances.union(current_columns)

# Convert common_appliances to a list
common_appliances = list(common_appliances)

# Select columns for training and testing
train_columns = common_appliances
test_columns = ['main']

# Create an empty dictionary to store DataFrames for each house
train_data_dict = {}

# Iterate over the files and read each one into the dictionary
for file in dataset_files[:-1]:  # Exclude the test file
    house_name = file.split('_')[1]  # Extract house name from file name
    train_data_dict[house_name] = pd.read_csv(os.path.join(dataset_folder, file))

# Filter data to include only selected columns
train_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file))[train_columns] for file in dataset_files[:-1]])
test_data = pd.read_csv(os.path.join(dataset_folder, dataset_files[-1]))[test_columns]

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
    existing_appliances = set(house_data.columns[1:])  # Get existing appliances in the house
    missing_appliances = appliance_colors.keys() - existing_appliances

    # Add missing appliances with zeros if needed
    if missing_appliances:
        house_data = house_data.reindex(columns=appliance_colors.keys(), fill_value=0)

    # Now you can safely calculate means without KeyError
    appliance_means = house_data[list(appliance_colors.keys())].mean().sort_values(ascending=False)
    main_mean = house_data['main'].mean()

    # Print statistics
    print(f"--- House {house_name} ---")
    print(f"Appliance Means:")
    print(appliance_means)
    print(f"Main Power Mean: {main_mean:.2f}")

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

# Ensure tight layout
plt.tight_layout()

# Display custom legend outside the subplots
plt.legend(handles=legend_handles, labels=legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1), title="Appliances", fancybox=True, shadow=True, ncol=1)

plt.show()
