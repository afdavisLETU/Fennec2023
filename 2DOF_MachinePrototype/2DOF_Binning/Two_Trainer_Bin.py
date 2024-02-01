import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Two_Load_Data_Bin import extractData

os.chdir('/home/coder/workspace/Data/2DOF_Machine/Ten_Binning_Data')

# List of CSV file paths
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
csv_list = [Data_Set_1,Data_Set_10]#,Data_Set_4,Data_Set_5,Data_Set_6,Data_Set_7,Data_Set_8,Data_Set_9,Data_Set_10]  


# # Define the neural network model
model = Sequential([
    LSTM(units=250, activation='tanh',return_sequences=False),
    Dense(units=75, activation='tanh'),
    Dense(units=50, activation='tanh'),
    Dense(units=2, activation='softmax')  
])


# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

inputs, outputs = extractData(csv_list)
# print(inputs[0])
print(outputs)
model.fit(inputs, outputs, epochs=3, batch_size=500)
# model.fit(inputs, outputs, epochs=5, batch_size=250)
model.fit(inputs, outputs, epochs=5, batch_size=1000)
model.fit(inputs, outputs, epochs=15, batch_size=2500)

# Save the model
model.save('2DOF-CGTenBin.h5')