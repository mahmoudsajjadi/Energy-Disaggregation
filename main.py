import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Load REDD dataset
# dataset_folder = '/home/seyedmahmouds/Projects/Energy Disaggregation/Energy-Disaggregation/Datasets/REDD/'
dataset_folder = './Datasets/REDD/'
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
redd_data = pd.concat([pd.read_csv(os.path.join(dataset_folder, file)) for file in dataset_files])



