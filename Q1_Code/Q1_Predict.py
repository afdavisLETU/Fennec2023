import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import RNN_model_predict

sim_model = 'Sim_Model.h5'
timesteps = 25
num_predictions = 20 * 50 #First number is lenght of prediction in seconds
pred_offset = 10 * 50 #First number is seconds offset
# Test Data
test_data = "Low_Wind/019_AA.xlsx"
print("Loading Data")

actual, predicted = RNN_model_predict(sim_model, test_data, timesteps, num_predictions, pred_offset)

deviation = np.std(actual[0:num_predictions])
mae = np.mean(np.abs(actual[0:num_predictions] - predicted[:]))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)


# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/50) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
plt.title("Motion Prediction")
plt.plot(x, actual[0:num_predictions], label='Actual')
plt.plot(x, predicted[:],'r--', label='Predicted')
plt.xlabel("Time (s)")
plt.ylabel("IMU[0]")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('SimPrediction.png')