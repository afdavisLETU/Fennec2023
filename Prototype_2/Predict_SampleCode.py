#Drone Motion Neural Network Prediciton
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import matplotlib.pyplot as plt
from DMM_DataLoader import FNN_model_predict, RNN_model_predict

model = 'DMNN.h5'
readfile = 'Test_AutoData.csv'
writefile = 'Prediction.csv'
timesteps = 10
num_predictions = 258 # If value error: reduce number # If shape error: increase to displayed number

actual, predicted = RNN_model_predict(model, readfile, writefile, timesteps, num_predictions)

# Generate x-axis values
x = range(num_predictions)

# Plotting the data
plt.plot(x, actual, label='Actual')
plt.plot(x, predicted,'r--', label='Predicted')

# Adding labels and title
plt.ylabel('Height')
plt.title('Drone Motion Prediction')
plt.legend()
plt.show()