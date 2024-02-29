import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Q1_DataLoader import csv_load_data

# Length of the input sequence during training
timesteps = 15
data_coeff = 1
output = 5
model_name = 'Model1.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
data_sets = 150

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
checkpoint = ModelCheckpoint('best_model.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')
early_stopping = EarlyStopping(monitor='loss', 
                               patience=2, 
                               verbose=1, 
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=5,
                              verbose=1,
                              min_lr=1e-6)
# Train the model
inputs, outputs = get_data(data_sets, data_coeff)
model.fit(inputs, outputs, epochs=1, batch_size=2500,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=5, batch_size=5000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=10, batch_size=15000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=1, batch_size=2500,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=5, batch_size=5000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=10, batch_size=15000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=25, batch_size=25000,callbacks=[checkpoint, early_stopping, reduce_lr])