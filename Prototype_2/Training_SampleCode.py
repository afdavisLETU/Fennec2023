#Drone Motion Neural Network Training with Automated Data Collection
#Written By: Micah Heikkila
#Last Modified: July 2,2023

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from DMM_DataLoader import FNN_load_data, RNN_load_data

timesteps = 10

# Define the neural network model
model = Sequential([
    GRU(units=50, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

# Train the model
#Data Set 1
inputs, outputs = RNN_load_data('AutoData-1.csv', timesteps)
model.fit(inputs, outputs, epochs=15, batch_size=25)

#Data Set 2
inputs, outputs = RNN_load_data('AutoData-2.csv', timesteps)
model.fit(inputs, outputs, epochs=20, batch_size=50)

#Data Set 3
inputs, outputs = RNN_load_data('AutoData-3.csv', timesteps)
model.fit(inputs, outputs, epochs=50, batch_size=150)

# Save the model
#model.save('DMNN.h5')

