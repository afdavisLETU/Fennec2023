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
dataSet11 = "RW-P2_Data1.csv"
dataSet12 = "RW-P2_Data2.csv"
dataSet13 = "RW-P2_Data3.csv"
dataSet14 = "RW-P2_Data4.csv"
dataSet15 = "RW-P2_Data5.csv"
dataSet16 = "RW-P2_Data6.csv"
dataSet17 = "RW-P2_Data7.csv"
dataSet18 = "RW-P2_Data8.csv"
dataSet19 = "RW-P2_Data9.csv"
dataSet20 = "RW-P2_Data10.csv"
dataSet21 = "LW-P2_Data1.csv"
dataSet22 = "LW-P2_Data2.csv"
dataSet23 = "LW-P2_Data3.csv"
dataSet24 = "LW-P2_Data4.csv"
dataSet25 = "LW-P2_Data5.csv"
dataSet26 = "LW-P2_Data6.csv"
dataSet27 = "LW-P2_Data7.csv"
dataSet28 = "LW-P2_Data8.csv"
dataSet29 = "LW-P2_Data9.csv"
dataSet30 = "LW-P2_Data10.csv"

data = [dataSet2,dataSet3,dataSet4,dataSet5,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9,
        dataSet11,dataSet12,dataSet13,dataSet14,dataSet15,dataSet16,dataSet17,dataSet18,dataSet19,
        dataSet21,dataSet22,dataSet23,dataSet24,dataSet25,dataSet26,dataSet27,dataSet28,dataSet29]

inputs, outputs = RNN_load_data(dataSet1, timesteps)

for dataSet in data:
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)

# Define the neural network model
model_cg = Sequential([
    GRU(units=25, activation='tanh',return_sequences=False),
    Dense(units=20, activation='tanh'),
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
model_cg.save('P2_CG_Model.h5')