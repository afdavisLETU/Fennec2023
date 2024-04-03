import os
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Q1_DataProcess import load_data

# Length of the input sequence during training
for out in range(10):
    timesteps = 15
    data_coeff = 1
    output = out
    model_name = f'Model_{output}.h5'
    print(model_name)
    os.chdir('/home/coder/workspace/Data/Real_Data/')

    # Data Sets
    dataSet1 = "031B.csv"
    dataSet2 = "032B.csv"
    dataSet3 = "033B.csv"
    dataSet4 = "035B.csv"
    dataSet5 = "051B.csv"
    dataSet6 = "053B.csv"
    dataSet7 = "054B.csv"
    dataSet8 = "055B.csv"
    dataSet9 = "056B.csv"
    dataSet10 = "057B.csv"
    dataSet11 = "059B.csv"
    dataSet12 = "060B.csv"
    dataSet13 = "061B.csv"
    dataSet14 = "063B.csv"

    data = [globals()[f"dataSet{i}"] for i in range(1, 15)]

    print("Data Extraction...")
    inputs, outputs = load_data(data[0], timesteps, data_coeff, output)
    print(data[0], "Loaded", end='\r')
    for dataSet in data[1:]:
        inputData, outputData = load_data(dataSet, timesteps, data_coeff, output)
        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
        print(dataSet, "Loaded", end='\r', flush=True)
    print()
    print("Data Length:", len(inputs))


    # Define the neural network model
    #model = keras.models.load_model(model_name)
    model = Sequential([
        GRU(units=40, activation='linear',return_sequences=True),
        Dropout(0.1),
        GRU(units=32, activation='linear',return_sequences=False),
        Dense(units=16, activation='linear'),
        Dense(units=1, activation='linear')  
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(model_name, 
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
                                min_lr=1e-3)

    # Train the model
    model.fit(inputs, outputs, epochs=4, batch_size=500,callbacks=[checkpoint, early_stopping, reduce_lr])
    model.fit(inputs, outputs, epochs=8, batch_size=1500,callbacks=[checkpoint, early_stopping, reduce_lr])
    model.fit(inputs, outputs, epochs=12, batch_size=5000,callbacks=[checkpoint, early_stopping, reduce_lr])