import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Q1_DataLoader import csv_load_data, iter_gen

# Length of the input sequence during training
timesteps = 15
data_coeff = 1
output = 2
model_name = 'AccZ_Model.h5'
IG_model_name = 'AccZ-IG_Model.h5'
# Data Sets
dataSet1 = "Low_Wind/005_AA.csv"
dataSet2 = "Low_Wind/007_AA.csv"
dataSet3 = "Low_Wind/008_AA.csv"
dataSet4 = "Low_Wind/014_AA.csv"
dataSet5 = "Low_Wind/015_AA.csv"
dataSet6 = "Low_Wind/019_AA.csv"
dataSet7 = "Low_Wind/022_AA.csv"
dataSet8 = "Low_Wind/023_AA.csv"

data = [dataSet1,dataSet2,dataSet3]#,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8]

#Data Loading
print("Data Extraction...")
inputs, outputs = csv_load_data(data[0], timesteps, data_coeff, output)
print(data[0], "Loaded")
for dataSet in data[1:]:
    inputData, outputData = csv_load_data(dataSet, timesteps, data_coeff, output)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
    print(dataSet, "Loaded")
print("Data Length:", len(inputs))

# Define the neural network model
model = Sequential([
    GRU(units=500, activation='tanh',return_sequences=False),
    Dense(units=1, activation='tanh')  
])

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model.fit(inputs, outputs, epochs=15, batch_size=1000)
model.fit(inputs, outputs, epochs=25, batch_size=2500)

# Save the model
model.save(model_name)
#model.save(IG_model_name)
print("Model Saved")

#Iterative Generalization
"""
for p in range(5):
    #IG Data Loading
    pred_prob = 5 + p*3 # Percent
    print("Data Extraction...")
    inputs, outputs = iter_gen(IG_model_name, data[0], timesteps, data_coeff, output, pred_prob)
    print(data[0], "Loaded")
    print("Data Length:", len(inputs))

    model.fit(inputs, outputs, epochs=15, batch_size=2500)
    
    #Save the Model
    model.save(IG_model_name)
    print("IG Model Saved")
"""