import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from Two_Load_Data import extractData

os.chdir('/home/coder/workspace/Data/2DOF_Machine/')

# List of CSV file paths
Data_Set_1 = '2DOF_0High_Hori_Data.csv' # Add your file paths here
Data_Set_2 = '2DOF_0High_Hori_Data2.csv'
Data_Set_3 = '2DOF_1High_Vert_Data.csv'
Data_Set_4 = '2DOF_Balanced_Data1.csv'
Data_Set_5 = '2DOF_Balanced_Data2.csv'
Data_Set_6 = '2DOF_0High_Horiz_Data_23-10-06_with_CG.csv'
Data_Set_7 = '2DOF_1High_Vert_Data_23-10-06_with_CG.csv'
Data_Set_8 = '2DOF_1High_Vert_Data1_23-10-06_with_CG.csv'
Data_Set_9 = '2DOF_Balanced_Data_23-10-06_with_CG.csv'
Data_Set_10 = '2DOF_Balanced_Data1_23-10-06_with_CG.csv'
Data_Set_11 = '2DOF_Balanced_Data2_23-10-06_with_CG.csv'
csv_list = [Data_Set_6,Data_Set_7,Data_Set_8,Data_Set_9,Data_Set_10]  


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
print('DONE 3')
model.fit(inputs, outputs, epochs=3, batch_size=1000)
model.fit(inputs, outputs, epochs=5, batch_size=2500)
model.fit(inputs, outputs, epochs=8, batch_size=4000)
model.fit(inputs, outputs, epochs=10, batch_size=5000)

# Save the model
model.save('2DOF-CGL1.h5')