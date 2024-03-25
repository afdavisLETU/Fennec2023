import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import csv_model_predict

output = 0
model = f'Recovery_{output}.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
test_data = "synthetic_305.csv"
timesteps = 25
freq = 20
num_predictions = 20 * freq
pred_offset = 25 * freq

actual, predicted = csv_model_predict(model, test_data, timesteps, output, num_predictions, pred_offset)

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
print(mae)
print(calculate_nrmse(actual,predicted))
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
plt.axvline(x=timesteps/freq, color='#FFBD82', linestyle=':')
plt.plot(x, actual, color='#3498DB', label='Actual Values')
plt.plot(x, predicted, linestyle='--', label='Predicted Values')
plt.xlabel("Time (s)")
plt.ylabel("X-Acc (Normalized)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Predict.png')