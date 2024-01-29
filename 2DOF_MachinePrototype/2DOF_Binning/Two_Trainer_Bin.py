import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Two_Load_Data_Bin import extractData

os.chdir('/home/coder/workspace/Data/2DOF_Machine/Ten_Binning_Data')

# List of CSV file paths
Data_Set_1 = '2DOF_Data1' # Add your file paths here
Data_Set_2 = '2DOF_Data2'
Data_Set_3 = '2DOF_Data3'
Data_Set_4 = '2DOF_Data4'
Data_Set_5 = '2DOF_Data5'
Data_Set_6 = '2DOF_Data6'
Data_Set_7 = '2DOF_Data7'
Data_Set_8 = '2DOF_Data8'
Data_Set_9 = '2DOF_Data9'
Data_Set_10 = '2DOF_Data10'
csv_list = [Data_Set_2,Data_Set_3,Data_Set_4,Data_Set_5,Data_Set_6,Data_Set_7,Data_Set_8,Data_Set_9,Data_Set_10]  


# # Define the neural network model
model = Sequential([
    LSTM(units=50, activation='tanh'),
    Dense(units=30, activation='relu'),
    Dense(units=10, activation='tanh'),
    Dense(units=3, activation='softmax')  
])


# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

inputs, outputs = extractData(csv_list)

model.fit(inputs, outputs, epochs=3, batch_size=1000)
model.fit(inputs, outputs, epochs=5, batch_size=2500)
model.fit(inputs, outputs, epochs=8, batch_size=4000)
model.fit(inputs, outputs, epochs=10, batch_size=5000)

# Save the model
model.save('2DOF-CGTenBin.h5')