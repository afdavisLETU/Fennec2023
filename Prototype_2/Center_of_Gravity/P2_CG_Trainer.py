from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from P2_CG_DataLoader import RNN_load_data

# Length of the input sequence during training
timesteps = 15

# Data Sets
dataSet1 = "P2_Data1.csv"
dataSet2 = "P2_Data2.csv"
dataSet3 = "P2_Data3.csv"
dataSet4 = "P2_Data5.csv"
dataSet5 = "P2_Data6.csv"

data = [dataSet2,dataSet3,dataSet4,dataSet5]

inputs, outputs = RNN_load_data(dataSet1, timesteps)

for dataSet in data:
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)

# Define the neural network model
model_cg = Sequential([
    GRU(units=150, activation='tanh',return_sequences=False),
    Dense(units=100, activation='relu'),
    Dense(units=75, activation='linear'),
    Dense(units=3, activation='sigmoid')  
])

# Compile the model
model_cg.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model_cg.fit(inputs, outputs, epochs=3, batch_size=250)
model_cg.fit(inputs, outputs, epochs=5, batch_size=500)
model_cg.fit(inputs, outputs, epochs=10, batch_size=1000)
model_cg.fit(inputs, outputs, epochs=25, batch_size=5000)


# Save the model
model_cg.save('P2_CG_Model.h5')