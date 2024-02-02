import os
import re
import smt as smt
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
torch.manual_seed(42)

# Base directory where the result_*.jl folders are located
base_dir = "C:/Users/49157/Desktop/Simulations"

# List to store data
data_list = []

# Loop through each folder in the base directory
for folder in os.listdir(base_dir):
    if folder.startswith('result_parameters_') and folder.endswith('.jl'):
        folder_path = os.path.join(base_dir, folder)

        # Extract parameters using string separation
        parts = folder[len("result_parameters_"):-len(".jl")].split('_')

        # Check if the length of parts is 11 or more
        if len(parts) >= 11:
            params = {
                'numberOfLoopsRepititions': int(parts[0]),
                'RestTime': float(parts[1]),
                'ReadPulseTime': float(parts[2]),
                'ReadPulseVoltage': float(parts[3]),
                'ReadPulseRampTime': float(parts[4]),
                'SetPulseNumber': int(parts[5]),
                'SetPulseTime': float(parts[6]),
                'SetPulseVoltage': float(parts[7]),
                'ResetPulseNumber': int(parts[8]),
                'ResetPulseTime': float(parts[9]),
                'ResetPulseVoltage': float(parts[10])
            }

            # Read iread.mat file
            try:
                iread_data = loadmat(os.path.join(folder_path, 'PulseResults.mat'))
                # print(f"Shape of I in {folder}: {iread_data['I'].shape}")
                # print(f"Shape of V in {folder}: {iread_data['V'].shape}")
                # print(f"Shape of t in {folder}: {iread_data['t'].shape}")
                # print(f"Shape of read_idx_filt in {folder}: {iread_data['read_idx_filt'].shape}")
                # print(iread_data )

                # Extracting the indices from read_idx_filt
                indices = iread_data['read_idx_filt'][0] - 1  # Assuming read_idx_filt is a 2D array with shape (1, n)

                params['I_selected'] = iread_data['I'][indices, 0]  # Correcting the indexing
                params['V_selected'] = iread_data['V'][indices, 0]  # Correcting the indexing
                params['t_selected'] = iread_data['t'][indices, 0]  # Correcting the indexing

                # If you still need the original arrays, you can store them as well:
                #params['I'] = iread_data['I'][0:2000]
                # params['t'] = iread_data['t']
                #params['V'] = iread_data['V'][0:2000]
                # params['read_idx_filt'] = indices

                data_list.append(params)

            except FileNotFoundError:
                print(f"iread.mat not found in {folder}")

            except Exception as e:
                print(f"Error reading {folder}/PulsResults.mat: {e}")
        else:
            print(f"Folder name {folder} doesn't match the expected pattern!")

print(f"Processed {len(data_list)} folders.")

X_list = []
y_list = []

for data in data_list:
    # Extracting first 1000 values of I_selected based on 'read_idx_filt' as the target
    # I_selected = data['I_selected'][:1500]
    I_selected = np.zeros(1000)
    I_selected = data['I_selected'][:1000]

    # I_selected = I_selected_full[:1000] if len(I_selected_full) > 1000 else I_selected_full

    # Collecting parameter values as features
    X_values = [
        data['numberOfLoopsRepititions'],
        data['RestTime'],
        data['ReadPulseTime'],
        data['ReadPulseVoltage'],
        data['ReadPulseRampTime'],
        data['SetPulseNumber'],
        data['SetPulseTime'],
        data['SetPulseVoltage'],
        data['ResetPulseNumber'],
        data['ResetPulseTime'],
        data['ResetPulseVoltage']
    ]

    X_list.append(X_values)
    y_list.append(I_selected)

# Converting lists to arrays for model training
X_ = np.array(X_list)
y_ = np.array(y_list)
"""
fig, axes = plt.subplots(1, 4, figsize=(18, 9))  # Increase the width of the figure

param_names_0 = [
    'numberOfLoopsRepititions',
    'RestTime',
    'ReadPulseTime',
    'ReadPulseVoltage',
]

for param in range(4):
    ax = axes[param]  # Use only one index here
    ax.hist(X_[:, param], bins=10, alpha=0.5)
    ax.set_title(param_names_0[param])
    ax.set_xlabel("value")
    ax.set_ylabel('Frequency')

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()
fig, axes = plt.subplots(1, 4, figsize=(18, 9))  # Increase the width of the figure

param_names_1 = [

    'ReadPulseRampTime',
    'SetPulseNumber',
    'SetPulseTime',
    'SetPulseVoltage'
]
for param in range(4):
    ax = axes[param]  # Use only one index here
    ax.hist(X_[:, param], bins=10, alpha=0.5)
    ax.set_title(param_names_1[param])
    ax.set_xlabel("value")
    ax.set_ylabel('Frequency')

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(18, 9))  # Increase the width of the figure

param_names_2 = [
    'ResetPulseNumber',
    'ResetPulseTime',
    'ResetPulseVoltage'
]
for param in range(3):
    ax = axes[param]
    ax.hist(X_[:, param], bins=10, alpha=0.5)
    ax.set_title(param_names_2[param])
    ax.set_xlabel("value")
    ax.set_ylabel('Frequency')

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()

for i in range(3):
    plt.plot(X_[i])
    plt.xlabel("number of parameters")
    plt.ylabel("Frequency")
    plt.title("Input parameters")
    plt.show()
    """
#Normalizing the data
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X_)
y = scaler_X.fit_transform(y_)

X = np.array(X)
y = np.array(y)
#from sklearn.preprocessing import MinMaxScaler
#m_scaler = MinMaxScaler()
import numpy as np
from scipy.stats import pearsonr




# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# fig,ax = plt.subplots(figsize=(10,15))
# plt.plot(X_train,y_train)
# plt.show()

"""
for i in range(5):
    plt.plot(X_train[i])
    plt.show()

for i in range(5):
    plt.plot(y_train[i])
    plt.show()
"""

# convert data to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# load data


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
