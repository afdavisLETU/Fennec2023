from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from P2_CG_DataLoader import RNN_load_data

timesteps = 15

# Data Sets
dataSet1 = "P2_Data1.csv"
dataSet2 = "P2_Data2.csv"
dataSet3 = "P2_Data3.csv"
dataSet4 = "P2_Data5.csv"
dataSet5 = "P2_Data6.csv"

data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5]

for dataSet in data:
    inputs = RNN_load_data(dataSet, timesteps)

    dataset = pd.read_csv(dataSet).to_numpy()
    length_data, _, width_data = inputs.shape
    outputData = np.zeros((length_data, 3))
    outputs = np.zeros((0, 3))
    # Check conditions based on the dataset name
    if 'rw' in dataSet:
        # Assign [0, 1, 0] to the second element of outputData
        outputData[:, 0] = 1
    elif 'lw' in dataSet:
        # Assign [1, 0, 0] to the first element of outputData
        outputData[:, 1] = 1
    else:
        # Assign [0, 0, 1] to the third element of outputData
        outputData[:, 1] = 1

    #inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
    
    #Troubleshooting
    print(outputs.shape)
    print(inputs.shape)

# Define the neural network model
model_y = Sequential([
    GRU(units=150, activation='tanh',return_sequences=False),
    Dense(units=100, activation='relu'),
    Dense(units=75, activation='linear'),
    Dense(units=1, activation='linear')  
])

model_p = Sequential([
    GRU(units=150, activation='tanh',return_sequences=False),
    Dense(units=100, activation='relu'),
    Dense(units=75, activation='relu'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model_y.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model_p.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
model_y.fit(inputs, outputs[:,0], epochs=3, batch_size=250)
model_y.fit(inputs, outputs[:,0], epochs=5, batch_size=500)
model_y.fit(inputs, outputs[:,0], epochs=10, batch_size=1000)
model_y.fit(inputs, outputs[:,0], epochs=25, batch_size=5000)

model_p.fit(inputs, outputs[:,1], epochs=3, batch_size=250)
model_p.fit(inputs, outputs[:,1], epochs=5, batch_size=500)
model_p.fit(inputs, outputs[:,1], epochs=10, batch_size=1000)
model_p.fit(inputs, outputs[:,1], epochs=25, batch_size=5000)

# Save the model
model_y.save('P2_CG_Model_Y.h5')
model_p.save('P2_CG_Model_P.h5')
