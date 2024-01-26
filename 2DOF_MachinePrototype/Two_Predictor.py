import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from Two_Load_Data import extractData

os.chdir('/home/coder/workspace/Data/2DOF_Machine/')

model = load_model('2DOF-CGL1.h5')
readfile = ['2DOF_Balanced_Data2_23-10-06_with_CG.csv']

inputs, outputs = extractData(readfile)
predicted = model.predict(inputs)

# Plotting the data
P0, P1, P2 = [], [], []
x = range(len(outputs))
for r in x:
    P0.append(predicted[r, 0])
    P1.append(predicted[r, 1])
    P2.append(predicted[r, 2])

P0_mean = np.mean(P0)
P1_mean = np.mean(P1)
P2_mean = np.mean(P2)

# Calculate moving averages
window_size = 1000  # You can adjust the window size as needed
P0_avg = np.convolve(P0, np.ones(window_size) / window_size, mode='valid')
P1_avg = np.convolve(P1, np.ones(window_size) / window_size, mode='valid')
P2_avg = np.convolve(P2, np.ones(window_size) / window_size, mode='valid')

fig, axes = plt.subplots(3, 1, figsize=(15, 8))

# Plot the first subplot with moving average
axes[0].plot(x[:len(P0_avg)], P0_avg, color='red', label="P0 Moving Avg")
axes[0].scatter(x, P0, color='blue', marker='o', alpha=0.5, label="P0")
axes[0].axhline(y=P0_mean, color='orange', linestyle='--', label="P0 Mean")
axes[0].set_xlabel("X-axis")
axes[0].set_ylabel("Classification")
axes[0].set_title("[1,0,0]")
axes[0].legend()

# Plot the second subplot with moving average
axes[1].plot(x[:len(P1_avg)], P1_avg, color='red', label="P1 Moving Avg")
axes[1].scatter(x, P1, color='blue', marker='o', alpha=0.5, label="P1")
axes[1].axhline(y=P1_mean, color='orange', linestyle='--', label="P1 Mean")
axes[1].set_xlabel("X-axis")
axes[1].set_ylabel("Classification")
axes[1].set_title("[0,1,0]")
axes[1].legend()

# Plot the third subplot with moving average
axes[2].plot(x[:len(P2_avg)], P2_avg, color='red', label="P2 Moving Avg")
axes[2].scatter(x, P2, color='blue', marker='o', alpha=0.5, label="P2")
axes[2].axhline(y=P2_mean, color='orange', linestyle='--', label="P2 Mean")
axes[2].set_xlabel("X-axis")
axes[2].set_ylabel("Classification")
axes[2].set_title("[0,0,1]")
axes[2].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Adding labels and title
plt.savefig("2DOFPredict.png") #this is super important on this system or else you won't be able to see plots
plt.show()

#IF YOU WANT TO OVERLAY THE INPUT DATA, GET THE CODE FROM P2_CG_Predictor.py