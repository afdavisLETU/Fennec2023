import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Q3_DataLoader import RNN_load_data

# Length of the input sequence during training
timesteps = 750
data_coeff = 0.25

# Data Sets
dataSet1 = "AA_Data1.xlsx"
dataSet2 = "BB_Data1.xlsx"
dataSet3 = "CC_Data1.xlsx"

data = [dataSet1,dataSet2,dataSet3]

inputs, outputs = RNN_load_data(data[0], timesteps, data_coeff)
for dataSet in data[1:]:
    print("Data Set Loaded")
    inputData, outputData = RNN_load_data(dataSet, timesteps, data_coeff)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
print("Data Loading Finished")
print("Data Length:", len(inputs))

# Define the neural network model
model_cg = Sequential([
    GRU(units=100, activation='tanh',return_sequences=True),
    GRU(units=75, activation='tanh',return_sequences=False),
    Dense(units=50, activation='tanh'),
    Dense(units=3, activation='softmax')  
])

# Compile the model
model_cg.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cg.fit(inputs, outputs, epochs=3, batch_size=250)
model_cg.fit(inputs, outputs, epochs=5, batch_size=500)

# Save the model
model_cg.save('CG_Model.h5')