import matplotlib.pyplot as plt
import numpy as np

# Algorithm names and corresponding metrics
algorithms = ['MLP', 'CNN', 'LSTM', 'CNN_LSTM', 'Bi_LSTM', 'VAE', 'GRUMLPNet']

mae = [1.1095, 0.0463, 0.0482, 0.0282, 0.1112, 0.5272, 0.005470]
mse = [0.88, 0.0055, 0.0078, 0.0026, 0.0359, 0.5452, 0.00064]
rmse = [1.4688, 0.0745, 0.0886, 0.0519, 0.1895, 0.7384, 0.008]
# Replace the extremely large value with a reasonable value (e.g., 10) for visualization
mape = [526.3307, 0.2828, 0.4055, 0.2067, 0.9968, 10, 2.1]

# Scale up test loss for better visibility

mae=[val*10 for val in mae]
mse=[val*10 for val in mse]
rmse=[val*10 for val in rmse]
N = 7
ind = np.arange(N)
width = 0.15


bar2 = plt.bar(ind + width, np.log1p(mae), width, color='g')
bar3 = plt.bar(ind + width * 2, np.log1p(mse), width, color='b')
bar4 = plt.bar(ind + width * 3, np.log1p(rmse), width, color='purple')
bar5 = plt.bar(ind + width * 4, np.log1p(mape), width, color='orange')

plt.xlabel("Algorithms")
plt.ylabel('Metrics (log scale)')
plt.title("Comparison of Deep Learning Models")

plt.xticks(ind + width * 2, algorithms)
plt.legend(( bar2, bar3, bar4, bar5), ('MAE', 'MSE', 'RMSE', 'MAPE'))
plt.show()
