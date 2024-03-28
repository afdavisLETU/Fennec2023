import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataProcess import model_simulation

os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
test_data = "synthetic_255.csv"
timesteps = 20
freq = 10
num_predictions = 20 * freq
pred_offset = 45 * freq

actual, predicted = model_simulation(test_data, timesteps, num_predictions, pred_offset)
print(actual)
print(predicted)
def calculate_nrmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    nrmse = rmse / (np.max(actual) - np.min(actual))
    return nrmse

for i in range(9):
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

fig, axs = plt.subplots(9, 1, figsize=(8, 20), dpi=300)

for i in range(9):
    axs[i].set_title("Motion Prediction: " + test_data)
    axs[i].axvline(x=timesteps/freq, color='#FFBD82', linestyle=':')
    axs[i].plot(x, actual[:,i], color='#3498DB', label='Actual Values')
    axs[i].plot(x, predicted[:,i], linestyle='--', label='Predicted Values')
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(f'Output {i}')
    axs[i].legend()
fig.tight_layout()

plt.savefig('Simulation.png')
plt.show()