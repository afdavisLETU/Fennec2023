#Drone Motion Neural Network Training with Automated Data Collection
#Written By: Micah Heikkila
#Last Modified: July 2,2023

from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from P2_DataLoader import RNN_load_data


timesteps = 25

# Data Sets
dataSet1 = "AutoP2_Data1.csv"
dataSet2 = "AutoP2_Data2.csv"
dataSet3 = "AutoP2_Data3.csv"
dataSet4 = "AutoP2_Data4.csv"
dataSet5 = "AutoP2_Data5.csv"
dataSet6 = "AutoP2_Data6.csv"
dataSet7 = "AutoP2_Data7.csv"
dataSet8 = "Manual_Data1_with_Zero.csv"
dataSet9 = "Manual_Data2_with_Zero.csv"
dataSet10 = "Manual_Data3_with_Zero.csv"
dataSet11 = "Manual_Data4_with_Zero.csv"
dataSet12 = "Manual_Data5_with_Zero.csv"
# dataSet13 = "data13.csv"
data = [dataSet2,dataSet3,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9,dataSet10,dataSet11]

# Timesteps
inputs, outputs = RNN_load_data(dataSet1, timesteps)
for dataSet in data: 
    # Collect Inputs/Outputs
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)

# Define the neural network model
model = Sequential([
    GRU(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=3, activation='linear')  
])

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model.fit(inputs, outputs, epochs=5, batch_size=250)
model.fit(inputs, outputs, epochs=5, batch_size=500)
model.fit(inputs, outputs, epochs=10, batch_size=1000)
model.fit(inputs, outputs, epochs=15, batch_size=2500)
model.fit(inputs, outputs, epochs=15, batch_size=5000)

# Save the model
model.save('P2_Model.h5')

