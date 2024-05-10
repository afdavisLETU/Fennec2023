import os
import numpy as np
import pandas as pd
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
    os.chdir('/home/coder/workspace/Data/Simulator_Data/')

    # Data Sets
    dataSet1 = "Flight1.csv"
    dataSet2 = "Flight2.csv"
    dataSet3 = "Flight3.csv"
    dataSet4 = "Flight4.csv"
    dataSet5 = "Flight5.csv"
    dataSet6 = "Flight6.csv"
    dataSet7 = "Flight7.csv"
    dataSet8 = "X_Data.csv"
    dataSet9 = "Y_Data.csv"
    dataSet10 = "Y_Data2.csv"
    dataSet11 = "Z_Data.csv"
    dataSet12 = "Yaw_Data.csv"
    dataSet13 = "Yaw_Data2.csv"
    dataSet14 = "Drift1.csv"
    dataSet15 = "Drift2.csv"
    dataSet16 = "Acro1.csv"
    dataSet17 = "Acro2.csv"
    dataSet18 = "Acro3.csv"
    dataSet19 = "Acro4.csv"
    dataSet20 = "Acro5.csv"



    data = [globals()[f"dataSet{i}"] for i in range(1, 16)]

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
    model.fit(inputs, outputs, epochs=6, batch_size=500,callbacks=[checkpoint, early_stopping, reduce_lr])
    model.fit(inputs, outputs, epochs=12, batch_size=1500,callbacks=[checkpoint, early_stopping, reduce_lr])
    model.fit(inputs, outputs, epochs=24, batch_size=5000,callbacks=[checkpoint, early_stopping, reduce_lr])
    #model.fit(inputs, outputs, epochs=36, batch_size=15000,callbacks=[checkpoint, early_stopping, reduce_lr])