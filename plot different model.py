import numpy as np
import matplotlib.pyplot as plt
import lstm
import mlp
import cnn
import util
import hybridnn
import bidirectional_lstm
import VAE
import mlp_gru
# Assuming predictions and actual data are numpy arrays
# Modify the following lines based on your actual variable names
predictions_lstm = lstm.predictions[0]
predictions_cnn = cnn.predictions[0]
#print(predictions_mlp)
predictions_mlp = mlp.predictions_test[0]
actual_data = util.y_test[0]
predictions_cnn_lstm=hybridnn.predictions[0]
bi_lstm=bidirectional_lstm.predictions[0]
mlp_gru=newmodel_final.pred
# Check the lengths of your data
print("Length of Actual Data:", len(actual_data))
print("Length of LSTM Predictions:", len(predictions_lstm))
print("Length of CNN Predictions:", len(predictions_cnn))
print("Length of MLP Predictions:", len(predictions_mlp))
print("Length of cnn_lstm Predictions:", len(predictions_cnn_lstm))
# Set up the subplot grid
fig, axs = plt.subplots(7, 1, figsize=(12, 8), sharex=True)

# Plot actual data
axs[0].plot(actual_data, label='Actual Data', color='blue')

# Plot predictions from LSTM model
axs[1].plot(predictions_lstm, label='LSTM Predictions', color='orange')

# Plot predictions from CNN model
axs[2].plot(predictions_cnn, label='CNN Predictions', color='green')

# Plot predictions from MLP model
axs[3].plot(predictions_mlp, label='MLP Predictions', color='red')
axs[4].plot(predictions_cnn_lstm,label="CNN_LSTM",color='purple')
axs[5].plot(bi_lstm,label="Bidirectional_LSTM",color='cyan')
axs[6].plot(mlp_gru,label="MLP_GRU",color="pink")
# Customize the plot appearance
for ax in axs:
    ax.legend()
    ax.grid(True)

# Set common labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.suptitle('Comparison of Actual Data and Model Predictions')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
