#Drone Motion Neural Network Prediciton
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import matplotlib.pyplot as plt
from P2_DataLoader import RNN_model_predict

model = 'P2_Model.h5'
readfile = 'Manual_Data5_with_Zero.csv'
writefile = 'Prediction.csv'
timesteps = 25
num_predictions = 9327 # If value error: reduce number # If shape error: increase to displayed number

actual, predicted = RNN_model_predict(model, readfile, writefile, timesteps, num_predictions)

# Generate x-axis values
x = range(num_predictions)

# Plotting the data
plt.plot(x, actual[0], label='Actual')
plt.plot(x, predicted[0],'r--', label='Predicted')

# Adding labels and title
plt.ylabel('Angle')
plt.title('Motion Prediction')
plt.legend()
plt.show()
plt.save('graph.png')