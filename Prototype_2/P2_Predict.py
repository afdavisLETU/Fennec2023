#Drone Motion Neural Network Prediciton
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import numpy as np
import matplotlib.pyplot as plt
from P2_DataLoader import RNN_model_predict

model_y = 'P2_Model_Y.h5'
model_p = 'P2_Model_P.h5'
readfile = 'P2_Data7.csv'
writefile = 'Prediction.csv'
timesteps = 15
num_predictions = 250 # If value error: reduce number # If shape error: increase to displayed number

actual, predicted = RNN_model_predict(model_y, model_p, readfile, writefile, timesteps, num_predictions)

deviation = np.std(actual[0:num_predictions,0])
mae = np.mean(np.abs(actual[0:num_predictions,0] - predicted[:,0]))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)
deviation = np.std(actual[0:num_predictions,1])
mae = np.mean(np.abs(actual[0:num_predictions,1] - predicted[:,1]))
accuracy = deviation / (deviation + mae)
print("Percent Accuracy:", accuracy*100)


# Generate x-axis values
x = []
for t in range(num_predictions):
    x.append(t/25)
x = np.array(x)

# Plotting the data
plt.subplot(2, 1, 1)
plt.title("Motion Prediction")
plt.plot(x, actual[0:num_predictions,0], label='Actual')
plt.plot(x, predicted[:,0],'r--', label='Predicted')
plt.xlabel("Time (s)")
plt.ylabel("Y Gyro")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, actual[0:num_predictions,1], label='Actual')
plt.plot(x, predicted[:,1],'r--', label='Predicted')
plt.xlabel("Time (s)")
plt.ylabel("P Gyro")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('MotionPrediction.png')