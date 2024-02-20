import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from Q3_DataLoader import RNN_load_data

# Length of the input sequence during training
timesteps = 750
data_coeff = 0.1

# Data Sets
dataSet1 = "Low_Wind/008_AA.csv"
dataSet2 = "Low_Wind/006_BB.csv"
dataSet3 = "Low_Wind/009_CC.csv"
dataSet4 = "Low_Wind/014_AA.csv"
dataSet5 = "Low_Wind/016_BB.csv"
dataSet6 = "Low_Wind/020_CC.csv"

data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5,dataSet6]

inputs, outputs = RNN_load_data(data[0], timesteps, data_coeff)
for dataSet in data[1:]:
    print(dataSet, "Loaded")
    inputData, outputData = RNN_load_data(dataSet, timesteps, data_coeff)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
print("Data Loading Finished")
print("Data Length:", len(inputs))

# Define the neural network model
model_cg = Sequential([
    GRU(units=128, activation='tanh', return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.2),
    GRU(units=64, activation='tanh', return_sequences=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=3, activation='softmax')  
])


# Compile the model
opt = Adam(learning_rate=0.001)
model_cg.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
model_cg.fit(inputs, outputs, epochs=2, batch_size=150)
model_cg.fit(inputs, outputs, epochs=3, batch_size=250)
#model_cg.fit(inputs, outputs, epochs=15, batch_size=750)

# Save the model
model_cg.save('CG_Model.h5')