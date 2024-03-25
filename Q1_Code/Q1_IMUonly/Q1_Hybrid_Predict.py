import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import hybrid_pred

os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
test_data = "synthetic_305.csv"
r_timesteps = 25
pred_timesteps = 10
output = 4
freq = 20
num_predictions = 15 * freq
pred_offset = 25 * freq
pred_model = f'Pred_{output}.h5'

print("Loading Data...")
actual, predicted, data = hybrid_pred(pred_model, test_data, r_timesteps, pred_timesteps, output, num_predictions, pred_offset)

def calculate_nrmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    nrmse = rmse / (np.max(actual) - np.min(actual))
    return nrmse

deviation = np.std(actual)
mae = np.mean(np.abs(actual - predicted))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)
print("NRMSE:", 100*calculate_nrmse(actual,predicted))


# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/freq) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
os.chdir('/home/coder/workspace/Graphs/')
plt.style.use("./styles/rose-pine.mplstyle")
plt.figure(dpi=300)
plt.title("Motion Prediction: " + test_data)
plt.axvline(x=r_timesteps/freq, color='#FFBD82', linestyle=':')
plt.plot(x, actual, color='#3498DB', label='Actual Values')
plt.plot(x, predicted, linestyle='--', label='Predicted Values')
plt.plot(x, data[:,(7+output)], linestyle=':', label='Recovery Values')
plt.xlabel("Time (s)")
plt.ylabel("X-Acc (Normalized)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Predict.png')