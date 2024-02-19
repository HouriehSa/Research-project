import util
import numpy as np
import torch.optim as optim
from torch.nn.modules import conv
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'


model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(11,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1000)  # No activation for regression output
        ])





model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
history = model.fit(util.X_train, util.y_train, epochs=500, batch_size=32, validation_data=(util.X_test, util.y_test))



train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plot the MSE loss
plt.plot(epochs, train_loss, label='Train MSE Loss')
plt.plot(epochs, val_loss, label='Validation MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training MSE Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
model_summary = Model(inputs=model.input, outputs=model.output)
print(model_summary.summary())
loss = model.evaluate(util.X_test, util.y_test)
print(f"Test MSE: {loss}")
predictions = model.predict(util.X_test, verbose=2)
mse_test = mean_squared_error(util.y_test, predictions)
mae_test = mean_absolute_error(util.y_test, predictions)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(util.y_test, predictions)

# Print the test metrics
print(f'Test MSE: {mse_test}')
print(f'Test MAE: {mae_test}')
print(f'Test RMSE: {rmse_test}')
print(f'Test MAPE: {mape_test}')
for i in range(10):
    plt.figure(figsize=(14, 6))

    # Plotting the true values
    plt.plot(util.y_test[i], label='True values', color='blue', alpha=0.7)

    # Plotting the predicted values
    plt.plot(predictions[i], label='Predictions', color='red', linestyle='--', alpha=0.7)

    plt.title('Comparison of True Values and Predictions')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('I_selected Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

