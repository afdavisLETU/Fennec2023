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
dataSet1 = "AutoP2_Data1.csv"
dataSet2 = "AutoP2_Data2.csv"
dataSet3 = "AutoP2_Data3.csv"
dataSet4 = "AutoP2_Data4.csv"
dataSet5 = "AutoP2_Data5.csv"
dataSet6 = "AutoP2_Data6.csv"
dataSet7 = "AutoP2_Data7.csv"
dataSet8 = "PS4_Data1.csv"
dataSet9 = "PS4_Data2.csv"
dataSet10 = "PS4_Data3.csv"
dataSet11 = "PS4_Data4.csv"
dataSet12 = "PS4_Data5.csv"
dataSet13 = "PS4_Data6.csv"
dataSet14 = "PS4_Data7.csv"
dataSet15 = "PS4_Data8.csv"
dataSet16 = "PS4_Data9.csv"
dataSet17 = "PS4_Data10.csv"
dataSet18 = "PS4_Data11.csv"
dataSet19 = "PS4_Data12.csv"
dataSet20 = "PS4_Data13.csv"
dataSet21 = "PS4_Data14.csv"

#Exclude dataSet1 in data array (Used later below)
data = [dataSet2,dataSet3,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9,dataSet10,dataSet11,dataSet12,
        dataSet13,dataSet14,dataSet15,dataSet16,dataSet17,dataSet18,dataSet19,dataSet21]

# Timesteps
inputs, outputs = RNN_load_data(dataSet1, timesteps)
for dataSet in data: 
    # Collect Inputs/Outputs
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)


# Define the neural network model
model_y = Sequential([
    GRU(units=500, activation='tanh',return_sequences=False),
    Dense(units=150, activation='relu'),
    Dense(units=50, activation='linear'),
    Dense(units=1, activation='linear')  
])

model_p = Sequential([
    GRU(units=500, activation='tanh',return_sequences=False),
    Dense(units=150, activation='relu'),
    Dense(units=50, activation='linear'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model_y.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model_p.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model_y.fit(inputs, outputs[:,0], epochs=3, batch_size=250)
model_y.fit(inputs, outputs[:,0], epochs=5, batch_size=500)
model_y.fit(inputs, outputs[:,0], epochs=10, batch_size=1000)
model_y.fit(inputs, outputs[:,0], epochs=15, batch_size=2500)

model_p.fit(inputs, outputs[:,1], epochs=3, batch_size=250)
model_p.fit(inputs, outputs[:,1], epochs=5, batch_size=500)
model_p.fit(inputs, outputs[:,1], epochs=10, batch_size=1000)
model_p.fit(inputs, outputs[:,1], epochs=15, batch_size=2500)

# Save the model
model_y.save('P2_Model_Y.h5')
model_p.save('P2_Model_P.h5')
