import util
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.optim as optim
from torch.nn.modules import conv
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'

# Define the TensorFlow model
class TF_MLP(tf.keras.Model):
    def __init__(self):
        super(TF_MLP, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(11,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1000)
        ])

    def call(self, x):
        return self.model(x)

# Instantiate the TensorFlow model
tf_model = TF_MLP()

# Specify the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
criterion = tf.keras.losses.MeanSquaredError()

# Training loop
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs}")
    print("-" * 50)
    total_loss = 0

    for batch, (x, y) in enumerate(util.train_loader):
        x = tf.convert_to_tensor(x.numpy(), dtype=tf.float32)
        y = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)

        with tf.device(device):
            with tf.GradientTape() as tape:
                output = tf_model(x)
                loss = criterion(y, output)

            gradients = tape.gradient(loss, tf_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tf_model.trainable_variables))

        total_loss += loss.numpy()

        if batch % 10 == 0:
            print("loss:", loss.numpy())

    train_losses.append(total_loss / len(util.train_loader))
# Prediction and Evaluation
tf_model.model.trainable = False

total_test_loss = 0
predictions_test = []

for x_test, y_test in util.test_loader:
    x_test = tf.convert_to_tensor(x_test.numpy(), dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test.numpy(), dtype=tf.float32)

    output_test = tf_model(x_test)
    predictions_test.append(output_test.numpy())

    test_loss = criterion(y_test, output_test)
    total_test_loss += test_loss.numpy()

# Flatten the predictions_test array
predictions_test = np.concatenate(predictions_test, axis=0)
# Flatten the ground truth data
y_true = np.concatenate([y.numpy() for _, y in util.test_loader], axis=0)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_true, predictions_test)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, predictions_test)
print(f'Mean Absolute Error (MAE): {mae}')
