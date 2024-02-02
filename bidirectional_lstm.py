import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model
import util
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# Reshape the input data to 3D
X_train = util.X_train.reshape(util.X_train.shape[0], util.X_train.shape[1], 1)
X_test = util.X_test.reshape(util.X_test.shape[0], util.X_test.shape[1], 1)
from torchsummary import summary
input_shape = (X_train.shape[1], 1)  # Shape determined by your reshaped input data
model = Sequential([
    tf.keras.layers.Bidirectional(LSTM(200, return_sequences=True), input_shape=input_shape),
    Dense(64, activation='tanh'),
    Bidirectional(LSTM(150)),
    Dense(128, activation='tanh'),
    Dense(256, activation='tanh'),
    Dense(1000),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


# fit network
history = model.fit(X_train, util.y_train, epochs=10, batch_size=32, validation_data=(X_test, util.y_test), verbose=2,
                    shuffle=False)

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

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
predictions = model.predict(X_test, verbose=2)
