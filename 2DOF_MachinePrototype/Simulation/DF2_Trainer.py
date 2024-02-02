#Drone Motion Neural Network Training with Automated Data Collection
#Written By: Micah Heikkila
#Last Modified: July 2,2023

from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from DF2_DataLoader import RNN_load_data

timesteps = 25

# Data Sets
dataSet1 = "2DF_Data1.csv"
dataSet2 = "2DF_Data2.csv"
dataSet3 = "2DF_Data3.csv"
dataSet4 = "2DF_Data4.csv"
dataSet5 = "2DF_Data5.csv"

#Exclude dataSet1 in data array (Used later below)
data = [dataSet1,dataSet2,dataSet3,dataSet4]

# Timesteps
inputs, outputs = RNN_load_data(data[0], timesteps)
for dataSet in data[1:]: 
    # Collect Inputs/Outputs
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)


# Define the neural network model
model_p = Sequential([
    GRU(units=500, activation='tanh',return_sequences=False),
    Dense(units=75, activation='relu'),
    Dense(units=50, activation='linear'),
    Dense(units=1, activation='linear')  
])

model_y = Sequential([
    GRU(units=500, activation='tanh',return_sequences=False),
    Dense(units=75, activation='relu'),
    Dense(units=50, activation='linear'),
    Dense(units=1, activation='linear')  
])


# Compile the model
model_p.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model_y.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model_p.fit(inputs, outputs[:,0], epochs=3, batch_size=250)
model_p.fit(inputs, outputs[:,0], epochs=5, batch_size=500)
model_p.fit(inputs, outputs[:,0], epochs=15, batch_size=1000)
model_p.fit(inputs, outputs[:,0], epochs=25, batch_size=2500)
model_p.fit(inputs, outputs[:,0], epochs=25, batch_size=5000)

model_y.fit(inputs, outputs[:,1], epochs=3, batch_size=250)
model_y.fit(inputs, outputs[:,1], epochs=5, batch_size=500)
model_y.fit(inputs, outputs[:,1], epochs=15, batch_size=1000)
model_y.fit(inputs, outputs[:,1], epochs=25, batch_size=2500)
model_y.fit(inputs, outputs[:,1], epochs=25, batch_size=5000)

# Save the model
model_p.save('2DF_Model_P.h5')
model_y.save('2DF_Model_Y.h5')