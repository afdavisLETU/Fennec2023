import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from Q3_DataLoader import RNN_load_data
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Length of the input sequence during training
timesteps = 250
data_coeff = 0.75

# Data Sets
dataSet1 = "Norm400Hz_005_AA.csv"
dataSet2 = "Norm400Hz_006_BB.csv"
dataSet3 = "Norm400Hz_007_AA.csv"
dataSet4 = "Norm400Hz_008_AA.csv"
dataSet5 = "Norm400Hz_009_CC.csv"
dataSet6 = "Norm400Hz_010_AA.csv"
dataSet7 = "Norm400Hz_014_AA.csv"
dataSet8 = "Norm400Hz_015_AA.csv"
dataSet9 = "Norm400Hz_016_BB.csv"
dataSet10 = "Norm400Hz_019_AA.csv"
dataSet11 = "Norm400Hz_020_CC.csv"
dataSet12 = "Norm400Hz_021_CC.csv"
dataSet13 = "Norm400Hz_022_AA.csv"
dataSet14 = "Norm400Hz_023_AA.csv"
dataSet15 = "Norm400Hz_025_AA.csv"
dataSet16 = "Norm400Hz_026_CC.csv"
dataSet17 = "Norm400Hz_027_CC.csv"
dataSet18 = "Norm400Hz_028_AA.csv"

# Number of Data Sets so far: 11A, 2B, 5C

#8,6,9,10,16,20
#21,23

data = [dataSet2, dataSet4, dataSet5, dataSet6, dataSet9, dataSet11]

# Define the neural network model
model_cg = Sequential([
    # GRU(units=128, activation='tanh', return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    # Dropout(0.2),
    GRU(units=10, activation='tanh', return_sequences=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.2),
    Dense(units=5, activation='relu'),
    Dense(units=3, activation='softmax')  
])

# Compile the model
opt = Adam(learning_rate=0.001)
model_cg.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='accuracy', patience=3)
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5)
model_checkpoint_acc = ModelCheckpoint('CG_Model.h5', save_best_only=True, monitor='accuracy', mode='max')

# Randomize the order of datasets
random.shuffle(data)

print(data[0], "Loaded")
inputs, outputs = RNN_load_data(data[0], timesteps, data_coeff)
for dataSet in data[1:]:
    print(dataSet, "Loaded")
    inputData, outputData = RNN_load_data(dataSet, timesteps, data_coeff)
    inputs = np.concatenate((inputs, inputData), axis=0)
    outputs = np.concatenate((outputs, outputData), axis=0)
print("Data Loading Finished")
print("Data Length:", len(inputs))

# Train the model
model_cg.fit(inputs, outputs, epochs=10, batch_size=100, callbacks=[early_stopping, lr_reduction, model_checkpoint_acc])
# model_cg.fit(inputs, outputs, epochs=5, batch_size=500)
#model_cg.fit(inputs, outputs, epochs=3, batch_size=150)

# Save the model (Unnecessary because the ModelCheckpoint callback already saves it)
# model_cg.save('CG_Model.h5')
print("Model Saved")