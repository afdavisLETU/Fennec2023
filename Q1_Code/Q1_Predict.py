import os
import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import csv_model_predict

model = 'Model1.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
test_data = "synthetic_125.csv"
timesteps = 25
output = 3
freq = 50
num_predictions = 10 * freq
pred_offset = 20 * freq

actual, predicted = csv_model_predict(model, test_data, timesteps, output, num_predictions, pred_offset)

deviation = np.std(actual)
mae = np.mean(np.abs(actual - predicted))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)

# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/freq) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
plt.title("Motion Prediction: " + test_data)
plt.plot(x, actual, label='Actual')
plt.plot(x, predicted, linestyle='--',color='#FF0000', label='Final Pred') #'#92c6e2'
plt.xlabel("Time (s)")
plt.ylabel(model[:-9])
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Predict.png')