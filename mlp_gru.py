
import os
import re
import smt as smt
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torchsummary import summary

torch.manual_seed(42)

# Base directory where the result_*.jl folders are located
base_dir =  "C:/Users/49157/Desktop/Simulations"

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
                indices = iread_data['read_idx_filt'][
                              0] - 1  # Assuming read_idx_filt is a 2D array with shape (1, n)

                params['I_selected'] = iread_data['I'][indices, 0]  # Correcting the indexing
                params['V_selected'] = iread_data['V'][indices, 0]  # Correcting the indexing
                params['t_selected'] = iread_data['t'][indices, 0]  # Correcting the indexing

                # If you still need the original arrays, you can store them as well:
                # params['I'] = iread_data['I'][0:2000]
                # params['t'] = iread_data['t']
                # params['V'] = iread_data['V'][0:2000]
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

# Normalizing the data
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X_)
y = scaler_X.fit_transform(y_)

X = np.array(X)
y = np.array(y)

import numpy as np
from scipy.stats import pearsonr

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# convert data to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# load data

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

from torch import nn
import torch
from tqdm import tqdm



from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self, h_dim_params, h_dim_rnn, n_layers=1):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(11, h_dim_params),
            nn.ReLU(inplace=False)
        )

        self.rnn = nn.GRU(h_dim_params + 1, h_dim_rnn, n_layers, batch_first=True)

        self.head_projection = nn.Sequential(
            nn.Linear(h_dim_rnn, h_dim_rnn),
            nn.ReLU(inplace=False),
            nn.Linear(h_dim_rnn, 1)
        )

        self.h_dim_rnn = h_dim_rnn
        self.h_dim_params = h_dim_params
        self.n_layers = n_layers

    def forward(self, params, seq: torch.Tensor):
        loss_fn = torch.nn.MSELoss()
        batch_size, seq_len = seq.shape
        h = torch.zeros(self.n_layers, batch_size, self.h_dim_rnn, device=seq.device)
        loss = 0
        phi = self.phi(params)
        for t in range(1, seq_len):
            x_in = torch.cat((phi, seq[:, t-1].reshape(1, 1)), 1).unsqueeze(1)
            _, h = self.rnn(x_in, h)
            next_output = self.head_projection(h)
            loss += loss_fn(next_output, seq[:, t])
        return loss / seq_len


    def fit(self, train_loader, test_loader, epochs, lr, device):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(device)
        self.train()

        for e in range(epochs):
            # h = torch.zeros(self.n_layers, train_loader.batch_size, self.h_dim_rnn, device=device)
            loop_train = tqdm(train_loader)
            loop_train.set_description(f"Epoch {e} out of {epochs}")
            for X, y in loop_train:
                optimizer.zero_grad(set_to_none=True)
                X, y = X.to(device), y.to(device)
                loss = self.forward(X, y)
                loss.backward(retain_graph=True)
                optimizer.step()
                loop_train.set_postfix(loss=loss.item())

            test_loss = torch.zeros(1, device=device)
            overal_loss = []

            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    _, seq = y.shape
                    mse_loss = self.forward(X, y)
                    test_loss += mse_loss / seq

            overal_loss.append(test_loss)
            print(np.mean(overal_loss))

    def predict(self, params, seq):
        pred = []
        batch_size, seq_len = seq.shape
        h = torch.zeros(self.n_layers, batch_size, self.h_dim_rnn, device=seq.device)
        phi = self.phi(params)
        for t in range(1, seq_len):
            x_in = torch.cat((phi, seq[:, t - 1].reshape(1, 1)), 1).unsqueeze(1)
            _, h = self.rnn(x_in, h)
            next_output = self.head_projection(h)
            pred.append(next_output.item())
        return pred

model = Model(h_dim_params=25, h_dim_rnn=30, n_layers=1)
print(model)
model.fit(train_loader,
          test_loader,
          epochs=1,
          lr=1e-3,
          device="cpu")



for i, (x, y) in enumerate(test_loader):
    break

print(y)
pred = model.predict(x, y)
print(len(pred))
print(model)

plt.plot(pred, label='Predictions', color="red", linestyle='--')
plt.plot(y[0], label='True values', color='blue')
plt.show()
# Access and print model parameters
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Size: {param.size()}")

# Selecting the corresponding portion of y_test



# model.evaluate(predictions, y_test)
#model.plot(y_test, predictions)
