#Drone Motion Neural Network Prediciton
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import numpy as np
import matplotlib.pyplot as plt
from P2_DataLoader import RNN_model_predict

model_y = 'P2_Model_Y.h5'
model_p = 'P2_Model_P.h5'
readfile = 'PS4_Data13.csv'
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
x = range(num_predictions)

# Plotting the data
plt.plot(x, actual[0:num_predictions,0], label='Actual')
plt.plot(x, predicted[:,0],'r--', label='Predicted')

# Adding labels and title
plt.ylabel('Angle')
plt.title('Motion Prediction')
plt.legend()
plt.show()
plt.savefig('graph.png')