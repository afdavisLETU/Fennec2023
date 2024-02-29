import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from Q3_DataLoader import RNN_load_data, trim_data_to_min_file_length
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Length of the input sequence during training
timesteps = 256
data_coeff = 1

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

# Categorize the datasets
datasets_AA = [dataSet1, dataSet3, dataSet4, dataSet6, dataSet7, dataSet8, dataSet10, dataSet13, dataSet14, dataSet15, dataSet18]
datasets_BB = [dataSet2, dataSet9]
datasets_CC = [dataSet5, dataSet11, dataSet12, dataSet16, dataSet17]

# Shuffle the datasets in each category so that the model does not always train off of the earliest flights
random.shuffle(datasets_AA)
random.shuffle(datasets_BB)
random.shuffle(datasets_CC)

# Find the smallest category
min_category_size = min(len(datasets_AA), len(datasets_BB), len(datasets_CC))

# Trim each category to the size of the smallest category
datasets_AA = datasets_AA[:min_category_size]
datasets_BB = datasets_BB[:min_category_size]
datasets_CC = datasets_CC[:min_category_size]

# Combine the data file names into categories
data_categories = [datasets_AA, datasets_BB, datasets_CC]

# Trim the data in each category to the minimum total length
dataSets = trim_data_to_min_file_length(data_categories)

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

# Define callbacks
early_stopping = EarlyStopping(monitor='accuracy', patience=3)
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5)
model_checkpoint_acc = ModelCheckpoint('CG_Model.h5', save_best_only=True, monitor='accuracy', mode='max', verbose=1)

print(dataSets[0][0], "Loaded")
inputs, outputs = RNN_load_data(dataSets[0][0], timesteps, data_coeff)

first_dataset_loaded = False
for category in dataSets:
    for dataSet in category:
        if not first_dataset_loaded:
            first_dataset_loaded = True
            continue
        print(dataSet, "Loaded")
        inputData, outputData = RNN_load_data(dataSet, timesteps, data_coeff)
        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
print("Data Loading Finished")
print("Data Length:", len(inputs))

# Train the model
model_cg.fit(inputs, outputs, epochs=10, batch_size=128, callbacks=[early_stopping, lr_reduction, model_checkpoint_acc])
# model_cg.fit(inputs, outputs, epochs=2, batch_size=64, callbacks=[early_stopping, lr_reduction, model_checkpoint_acc])
# model_cg.fit(inputs, outputs, epochs=2, batch_size=32, callbacks=[early_stopping, lr_reduction, model_checkpoint_acc])
# model_cg.fit(inputs, outputs, epochs=5, batch_size=500)
#model_cg.fit(inputs, outputs, epochs=3, batch_size=150)

# Save the model (Unnecessary because the ModelCheckpoint callback already saves it)
# model_cg.save('CG_Model.h5')
print("Model Saved")