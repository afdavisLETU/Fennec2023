import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from Q3_DataLoader import RNN_load_data


# Length of the input sequence during training
timesteps = 500
data_coeff = 0.75

# Data Sets
dataSet1 = "Norm400Hz_008_AA.csv"
dataSet2 = "Norm400Hz_006_BB.csv"
dataSet3 = "Norm400Hz_009_CC.csv"
dataSet4 = "Norm400Hz_010_AA.csv"
dataSet5 = "Norm400Hz_016_BB.csv"
dataSet6 = "Norm400Hz_020_CC.csv"
dataSet7 = "Norm400Hz_021_CC.csv"
dataSet8 = "Norm400Hz_023_AA.csv"


data = [dataSet1,dataSet2,dataSet3]#,dataSet4,dataSet5,dataSet6,dataSet8,dataSet2]#,dataSet8]

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
model_cg.fit(inputs, outputs, epochs=5, batch_size=200)
#model_cg.fit(inputs, outputs, epochs=3, batch_size=150)

# Save the model
model_cg.save('CG_Model.h5')