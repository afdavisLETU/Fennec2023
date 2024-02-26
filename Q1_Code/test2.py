import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Q1_DataLoader import get_data, Micah_Filter
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
data_file = "synthetic_100.xlsx"
IMU, RCOU = get_data(data_file)
clean_IMU = Micah_Filter(IMU, 25, 100, 50)
points = 10000

# Generate x-axis values
x = []
for t in range(len(IMU)):
    x.append(t/50) #Set Denominator to Frequency of Data
x = np.array(x)

# Plotting the data
plt.title("Motion Prediction: " + data_file)
plt.plot(x[:points], IMU[:points,2], label='Actual')
plt.plot(x[:points], clean_IMU[:points,2],'r--', label='Predicted')
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Filter.png')
