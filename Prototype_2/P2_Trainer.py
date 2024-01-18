#Drone Motion Neural Network Training with Automated Data Collection
#Written By: Micah Heikkila
#Last Modified: July 2,2023

from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from P2_DataLoader import RNN_load_data

timesteps = 15

# Data Sets
dataSet1 = "P2_Data1.csv"
dataSet2 = "P2_Data2.csv"
dataSet3 = "P2_Data3.csv"


#Exclude dataSet1 in data array (Used later below)
data = [dataSet2,dataSet3]

# Timesteps
inputs, outputs = RNN_load_data(dataSet1, timesteps)
for dataSet in data: 
    # Collect Inputs/Outputs
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)


# Define the neural network model
model_y = Sequential([
    GRU(units=150, activation='tanh',return_sequences=False),
    Dense(units=100, activation='relu'),
    Dense(units=75, activation='linear'),
    Dense(units=1, activation='linear')  
])

model_p = Sequential([
    GRU(units=150, activation='tanh',return_sequences=False),
    Dense(units=100, activation='relu'),
    Dense(units=75, activation='relu'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model_y.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model_p.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model_y.fit(inputs, outputs[:,0], epochs=3, batch_size=250)
model_y.fit(inputs, outputs[:,0], epochs=5, batch_size=500)
model_y.fit(inputs, outputs[:,0], epochs=10, batch_size=1000)
model_y.fit(inputs, outputs[:,0], epochs=25, batch_size=5000)

model_p.fit(inputs, outputs[:,1], epochs=3, batch_size=250)
model_p.fit(inputs, outputs[:,1], epochs=5, batch_size=500)
model_p.fit(inputs, outputs[:,1], epochs=10, batch_size=1000)
model_p.fit(inputs, outputs[:,1], epochs=25, batch_size=5000)

# Save the model
model_y.save('P2_Model_Y.h5')
model_p.save('P2_Model_P.h5')
