import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Q1_DataLoader import csv_load_data

# Length of the input sequence during training
timesteps = 25
data_coeff = 0.5
output = 3
model_name = 'Model1.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
data_sets = 100

def get_data(data_sets, data_coeff):
    data = []
    for data_num in range(1,data_sets+1):
        data.append(f'synthetic_{data_num:03d}.csv')
    print("Data Extraction...")
    inputs, outputs = csv_load_data(data[0], timesteps, data_coeff, output)
    print(data[0], "Loaded", end='\r')
    for dataSet in data[1:]:
        inputData, outputData = csv_load_data(dataSet, timesteps, data_coeff, output)
        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
        print(dataSet, "Loaded", end='\r', flush=True)
    print()
    print("Data Length:", len(inputs))
    return inputs, outputs

# Define the neural network model
model = Sequential([
    GRU(units=20, activation='linear',return_sequences=True),
    GRU(units=30, activation='linear',return_sequences=False),
    Dense(units=15, activation='linear'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
inputs, outputs = get_data(data_sets, data_coeff)
model.fit(inputs, outputs, epochs=5, batch_size=2000)
model.fit(inputs, outputs, epochs=15, batch_size=5000)
#model.fit(inputs, outputs, epochs=30, batch_size=10000)
#model.fit(inputs, outputs, epochs=50, batch_size=50000)
# Save the model
model.save(model_name)
print("Model Saved")