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



# Specify the test house number
test_house_number = 6

# Load REDD dataset
# dataset_folder = '/home/seyedmahmouds/Projects/Energy Disaggregation/Energy-Disaggregation/Datasets/REDD/'
dataset_folder = './Datasets/REDD/'
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
dataset_files = natsorted(dataset_files)
redd_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files])


# Iterate over the remaining files and update common appliances
common_appliances = set(pd.read_csv(os.path.join(dataset_folder, dataset_files[0]), nrows=1).columns[1:])
all_appliances = set(pd.read_csv(os.path.join(dataset_folder, dataset_files[0]), nrows=1).columns[1:])
for file in dataset_files[1:]:
    file_path = os.path.join(dataset_folder, file)
    current_columns = set(pd.read_csv(file_path, nrows=1).columns[1:])
    # print(file, current_columns)
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
    # print(house_name)
    train_data_dict[house_name] = pd.read_csv(os.path.join(dataset_folder, file))



# Filter data to include only selected columns
train_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file))[train_columns] for file in dataset_files[:-1]])
test_data = pd.read_csv(os.path.join(dataset_folder, dataset_files[-1]))[test_columns]


plt.plot(train_data[common_appliances[0]])
# plt.plot(test_data['main'])
plt.xlabel('Time')
plt.ylabel('Main Power Consumption')
plt.title('Main Power Consumption Over Time')
plt.show()


# # Select House for Test and Train
# test_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files if f'house{test_house_number}' in file])
# train_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files if f'house{test_house_number}' not in file])


# # Standardize the data using StandardScaler
# scaler = StandardScaler()

# # Fit on training data
# scaler.fit(train_data.iloc[:, 1:])

# # Transform both training and test data
# train_data.iloc[:, 1:] = scaler.transform(train_data.iloc[:, 1:])
# test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])