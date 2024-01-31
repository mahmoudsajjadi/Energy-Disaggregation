# import gzip
# import tarfile

# # Relative paths from the "Energy Disaggregation" folder
# tar_gz_file = 'Datasets/PECAN_St/1minute_data_newyork.tar.gz'

# # Extract .tar.gz file
# with gzip.open(tar_gz_file, 'rb') as f_in:
#     with open(tar_gz_file[:-3], 'wb') as f_out:  # Removing the .gz extension
#         f_out.write(f_in.read())

# # Decompress the .tar file
# with tarfile.open(tar_gz_file[:-3], 'r') as tar:
#     tar.extractall()
    
    
# import tarfile

# # Relative paths from the "Energy Disaggregation" folder
# tar_file = 'Datasets/PECAN_St/1minute_data_newyork.tar'

# # Decompress .tar file
# with tarfile.open(tar_file, 'r') as tar:
#     tar.extractall()



import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_csv_data(csv_file_path):
    """
    Reads a CSV file, creates X and Y data with target as features and sum of EV divided by sum of X for each home,
    saves the preprocessed data as a CSV file, and splits them into training and testing sets.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # Read CSV file
    df = pd.read_csv(csv_file_path, nrows=100000)
    

    df['total_EV'] = df.groupby('dataid')['electricity'].transform('sum')
    df['sum_X'] = df.drop(['dataid', 'electricity', 'total_EV'], axis=1).sum(axis=1)
    df['target'] = df['total_EV'] / df['sum_X']

    output_csv = 'preprocessed_data.csv'
    df.to_csv(output_csv, index=False)  # Specify filename and avoid saving index

    X = df[['dataid', 'electricity']]

    # Target
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


csv_file_path = 'Datasets/PECAN_St/1minute_data_newyork/1minute_data_newyork.csv'

df = pd.read_csv('Datasets/PECAN_St/1minute_data_newyork/1minute_data_newyork.csv', nrows=100000)

date = '2019-06-24'

df['localminute'] = pd.to_datetime(df['localminute'])

df = df[df['localminute'].dt.date == pd.to_datetime('2019-06-24').date()]

home_consumption = df.drop(['leg1v', 'leg2v'], axis=1).groupby('dataid').sum().reset_index()


unique_dataids = df['dataid'].unique()

dataid_lengths = {}

for dataid in unique_dataids:
    length = len(df[df['dataid'] == dataid])
    dataid_lengths[dataid] = length

for dataid, length in dataid_lengths.items():
    print(f"dataid: {dataid}, length: {length}")

print(home_consumption)



time_resolution = 60


import matplotlib.pyplot as plt

num_rows = 2
num_cols = 5

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 8))

appliance_colors = {
    'grid': 'blue',
    'solar': 'green',
    'garage1': 'red',
    'heater1': 'orange',
    'kitchenapp1': 'purple',
    'kitchenapp2': 'cyan',
    'range1': 'yellow',
    'wellpump1': 'brown',
    'waterheater1': 'black',
    'freezer1': 'cyan',
    'air1': 'magenta',
    'air2': 'lime',
    'dishwasher1': 'pink',
    'refrigerator1': 'teal',
    'lights_plugs1': 'gold',
    'lights_plugs2': 'silver',
    'lights_plugs3': 'olive',
    'lights_plugs4': 'indigo',
    'utilityroom1': 'darkred',
    'kitchen1': 'darkgreen',
    'furnace1': 'navy',
    'furnace2': 'coral',
    'drye1': 'sienna',
    'clotheswasher1': 'darkblue',
    'livingroom1': 'lightgreen',
    'diningroom1': 'maroon',  
}


for i, dataid in enumerate(home_consumption['dataid'][:10]):  # Assuming you want to plot the first 10 dataids
    row_dataid = home_consumption[home_consumption['dataid'] == dataid].iloc[0]

    non_zero_values = row_dataid[row_dataid != 0][1:] / time_resolution
    threshold = non_zero_values['grid'] / 50

    filtered_values = non_zero_values[non_zero_values > threshold]
    
    if 'solar' not in filtered_values.index:
        filtered_values['solar'] = 0

    sorted_index = filtered_values.index.tolist()
    sorted_index.remove('grid')
    sorted_index.remove('solar')
    sorted_index = ['grid', 'solar'] + sorted(sorted_index, reverse=True)
    filtered_values_sorted = filtered_values.loc[sorted_index]

    filtered_values_sorted.loc[['grid', 'solar']] = filtered_values_sorted.loc[['grid', 'solar']].abs()
    filtered_values_sorted.loc[~filtered_values_sorted.index.isin(['grid', 'solar'])] *= -1

    colors = [appliance_colors[app] for app in filtered_values_sorted.index]

    row = i // num_cols
    col = i % num_cols
    bars = axs[row, col].bar(filtered_values_sorted.index, filtered_values_sorted,
                             color=colors)
    axs[row, col].set_title(f'home {dataid}')
    if col == 0:  
        axs[row, col].set_ylabel('Energy Consumption (kwh)')

    legend_labels = filtered_values_sorted.index
    axs[row, col].legend(bars, legend_labels, loc='upper right')

    axs[row, col].set_xticks([])

plt.tight_layout()
plt.show()



