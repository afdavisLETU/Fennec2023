import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataProcess import model_simulation

os.chdir('/home/coder/workspace/Data/Real_Data/')
test_data = "064B.csv"
timesteps = 15
freq = 20
num_predictions = 15 * freq
pred_offset = 0 * freq

actual, predicted = model_simulation(test_data, timesteps, num_predictions, pred_offset)

def calculate_nrmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    nrmse = rmse / (np.max(actual) - np.min(actual))
    return nrmse

for i in range(10):
    print(f'Model{i}:')
    print("Percent Accuracy:", (1-calculate_nrmse(actual[timesteps:,i],predicted[timesteps:,i]))*100)
    print("NRMSE:", calculate_nrmse(actual[timesteps:,i],predicted[timesteps:,i])*100)
# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/freq) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
os.chdir('/home/coder/workspace/Graphs/')
plt.style.use("./styles/rose-pine.mplstyle")

fig, axs = plt.subplots(10, 1, figsize=(8, 20), dpi=300)

for i in range(10):
    axs[i].set_title("Motion Prediction: " + test_data)
    axs[i].axvline(x=timesteps/freq, color='#FFBD82', linestyle=':')
    axs[i].plot(x, actual[:,i], color='#3498DB', label='Actual Values')
    axs[i].plot(x, predicted[:,i], linestyle='--', label='Predicted Values')
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(f'Output {i}')
    axs[i].legend()
fig.tight_layout()

plt.savefig('Final_Results.jpg')
plt.show()