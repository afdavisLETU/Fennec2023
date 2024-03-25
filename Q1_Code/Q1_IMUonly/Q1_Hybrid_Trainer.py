import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow warnings
from Q1_DataLoader import hybrid_train_data
import warnings
warnings.filterwarnings("ignore", message="Layer .* will not use cuDNN kernels.*")

# Length of the input sequence during training
pred_timesteps = 10
r_timesteps = 25
data_coeff = 1
output = 3
pred_model_name = f'Pred_{output}.h5'
recovery_model_name = f'Recovery_{output}.h5'
os.chdir('/home/coder/workspace/Data/Synthetic_Data/')
data_sets = 75

def get_data(data_sets, data_coeff):
    data = []
    for data_num in range(1,data_sets+1):
        data.append(f'synthetic_{data_num:03d}.csv')
    print("Data Extraction...")
    inputs, outputs = hybrid_train_data(recovery_model_name, data[0], r_timesteps, pred_timesteps, data_coeff, output)
    print(data[0], "Loaded", end='\r')
    for dataSet in data[1:]:
        inputData, outputData = hybrid_train_data(recovery_model_name, dataSet, r_timesteps, pred_timesteps, data_coeff, output)
        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
        print(dataSet, "Loaded", end='\r', flush=True)
    print()
    print("Data Length:", len(inputs))
    return inputs, outputs

# Define the neural network model
model = Sequential([
    GRU(units=50, activation='linear',return_sequences=True),
    GRU(units=50, activation='linear', return_sequences=False),
    Dense(units=25, activation='linear'),
    Dense(units=1, activation='linear')  
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.005), loss='mse', metrics=['accuracy'])
checkpoint = ModelCheckpoint(pred_model_name, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')
early_stopping = EarlyStopping(monitor='loss', 
                               patience=3, 
                               verbose=1, 
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.25,
                              patience=2,
                              verbose=0,
                              min_lr=1e-6)
# Train the model
inputs, outputs = get_data(data_sets, data_coeff)
model.fit(inputs, outputs, epochs=1, batch_size=2500,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=5, batch_size=5000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=10, batch_size=15000,callbacks=[checkpoint, early_stopping, reduce_lr])
model.fit(inputs, outputs, epochs=25, batch_size=25000,callbacks=[checkpoint, early_stopping, reduce_lr])