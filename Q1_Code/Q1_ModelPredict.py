import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataProcess import model_predict

output = 7
model = f'Model_{output}.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
test_data = "synthetic_251.csv"
timesteps = 20
freq = 20
num_predictions = 50 * freq
pred_offset = 5 * freq

actual, predicted = model_predict(model, test_data, timesteps, output, num_predictions, pred_offset)

def calculate_nrmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    nrmse = rmse / (np.max(actual) - np.min(actual))
    return nrmse

print("Percent Accuracy:", (1-calculate_nrmse(actual,predicted))*100)
print("NRMSE:", calculate_nrmse(actual,predicted)*100)
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
plt.ylabel(f'Output {output}')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Predict.png')