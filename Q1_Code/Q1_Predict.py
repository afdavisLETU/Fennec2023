import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import csv_model_predict

model = 'AccZ_Model.h5'
IG_model = 'AccZ-IG_Model.h5'
timesteps = 15
output = 2
num_predictions = 25*50 #First number is length of prediction in seconds
pred_offset = 5 * 50 #First number is seconds from start offset

# Test Data
test_data = "Low_Wind/010_AA.csv"
print("Loading Data")

actual, predicted = csv_model_predict(model, test_data, timesteps, output, num_predictions, pred_offset)

deviation = np.std(actual)
mae = np.mean(np.abs(actual - predicted))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)
"""
actual, predicted_IG = csv_model_predict(IG_model, test_data, timesteps, output, num_predictions, pred_offset)

deviation = np.std(actual)
mae = np.mean(np.abs(actual - predicted_IG))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)
"""

# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/50) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
plt.title("Motion Prediction: " + test_data)
plt.plot(x, actual, label='Actual')
plt.plot(x, predicted,'r--', label='Predicted')
#plt.plot(x, predicted_IG,'g--', label='Predicted')
plt.xlabel("Time (s)")
plt.ylabel(model[:-9])
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('SimPrediction.png')