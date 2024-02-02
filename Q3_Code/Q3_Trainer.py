from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from P2_CG_DataLoader import RNN_load_data

# Length of the input sequence during training
timesteps = 150

# Data Sets
dataSet1 = "P2_Data1.csv"
dataSet2 = "P2_Data2.csv"
dataSet3 = "P2_Data3.csv"
dataSet4 = "P2_Data4.csv"
dataSet5 = "P2_Data5.csv"
dataSet6 = "P2_Data6.csv"
dataSet7 = "P2_Data7.csv"
dataSet8 = "P2_Data8.csv"
dataSet9 = "P2_Data9.csv"
dataSet10 = "P2_Data10.csv"

data = [dataSet2,dataSet3,dataSet4,dataSet5,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9]

inputs, outputs = RNN_load_data(data[0], timesteps)

for dataSet in data[1:]:
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)

# Define the neural network model
model_cg = Sequential([
    GRU(units=25, activation='tanh',return_sequences=True),
    GRU(units=20, activation='tanh',return_sequences=False),
    Dense(units=10, activation='tanh'),
    Dense(units=3, activation='softmax')  
])

# Compile the model
model_cg.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cg.fit(inputs, outputs, epochs=3, batch_size=250)
model_cg.fit(inputs, outputs, epochs=5, batch_size=500)
model_cg.fit(inputs, outputs, epochs=10, batch_size=1000)
model_cg.fit(inputs, outputs, epochs=15, batch_size=5000)

# Save the model
model_cg.save('CG_Model.h5')