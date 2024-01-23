import numpy as np
import matplotlib.pyplot as plt
from P2_CG_DataLoader import RNN_model_predict

model_cg = 'P2_CG_Model.h5'
readfile = 'P2_Data7.csv'
writefile = 'Prediction_CG.csv'

# Length of the input sequence during training
timesteps = 15
# Allows you to predict on the first n number of data points for debugging
num_predictions = 250 # If value error: reduce number # If shape error: increase to displayed number

actual, predicted = RNN_model_predict(model_cg, readfile, writefile, timesteps, num_predictions)

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