from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from JK_Loader import RNN_load_data

# Length of the input sequence during training
timesteps = 150

# Data Sets
Data_Set_1 = '2DOF_Data1Z.csv' # Add your file paths here
Data_Set_2 = '2DOF_Data2X.csv'
Data_Set_3 = '2DOF_Data3.csv'
Data_Set_4 = '2DOF_Data4.csv'
Data_Set_5 = '2DOF_Data5.csv'
Data_Set_6 = '2DOF_Data6.csv'
Data_Set_7 = '2DOF_Data7.csv'
Data_Set_8 = '2DOF_Data8.csv'
Data_Set_9 = '2DOF_Data9.csv'
Data_Set_10 = '2DOF_Data10.csv'
data = [Data_Set_1,Data_Set_10]#,Data_Set_4,Data_Set_5,Data_Set_6,Data_Set_7,Data_Set_8,Data_Set_9,Data_Set_10]  

inputs, outputs = RNN_load_data(Data_Set_1, timesteps) 

for dataSet in data:
    inputData, outputData = RNN_load_data(dataSet, timesteps)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)

# Define the neural network model
model_cg = Sequential([
    GRU(units=25, activation='tanh',return_sequences=False),
    Dense(units=20, activation='tanh'),
    Dense(units=10, activation='tanh'),
    Dense(units=2, activation='softmax')  
])

# Compile the model
model_cg.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cg.fit(inputs, outputs, epochs=3, batch_size=250)
model_cg.fit(inputs, outputs, epochs=5, batch_size=500)
model_cg.fit(inputs, outputs, epochs=10, batch_size=1000)
model_cg.fit(inputs, outputs, epochs=15, batch_size=5000)

# Save the model
model_cg.save('JK_CG_Model.h5')