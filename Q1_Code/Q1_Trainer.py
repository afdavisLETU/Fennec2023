import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Q1_DataLoader import RNN_load_data

# Length of the input sequence during training
timesteps = 25
data_coeff = 1

# Data Sets
dataSet1 = "Low_Wind/005_AA.xlsx"
dataSet2 = "Low_Wind/007_AA.xlsx"
dataSet3 = "Low_Wind/008_AA.xlsx"
dataSet4 = "Low_Wind/014_AA.xlsx"
dataSet5 = "Low_Wind/015_AA.xlsx"
dataSet6 = "Low_Wind/019_AA.xlsx"
dataSet7 = "Low_Wind/022_AA.xlsx"
dataSet8 = "Low_Wind/023_AA.xlsx"

data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8]

print("Data Extraction...")
inputs, outputs = RNN_load_data(data[0], timesteps, data_coeff)
print(data[0], "Loaded")
for dataSet in data[1:]:
    inputData, outputData = RNN_load_data(dataSet, timesteps, data_coeff)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
    print(dataSet, "Loaded")
print("Data Length:", len(inputs))

# Define the neural network model
model_sim = Sequential([
    SimpleRNN(units=50, activation='relu',return_sequences=False),
    Dense(units=50, activation='relu'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model_sim.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
#model_sim.fit(inputs, outputs, epochs=5, batch_size=250)
model_sim.fit(inputs, outputs, epochs=15, batch_size=100)
model_sim.fit(inputs, outputs, epochs=25, batch_size=250)
model_sim.fit(inputs, outputs, epochs=50, batch_size=500)
#model_sim.fit(inputs, outputs, epochs=50, batch_size=1000)


# Save the model
model_sim.save('Sim_Model.h5')