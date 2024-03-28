import os
import pandas as pd
from tensorflow.keras.models import load_model
from Q3_DataLoader import RNN_load_data, trim_data_to_min_file_length
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

os.chdir('/home/coder/workspace/Data/Becky_Data/')

# Load the previously trained model
model_cg = load_model('CG_Model.h5')

# Define callbacks
early_stopping = EarlyStopping(monitor='accuracy', patience=3)
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=1, verbose=1, factor=0.5)
model_checkpoint_acc = ModelCheckpoint('CG_Model.h5', save_best_only=True, monitor='accuracy', mode='max', verbose=1)

# # Load the inputs and outputs from CSV files
# # Load the inputs from the CSV file and reshape it to the original shape
# inputs = pd.read_csv('Temp_inputs.csv').values.reshape(-1, 64, 11)
# outputs = pd.read_csv('Temp_outputs.csv').values

# Continue training the model
model_cg.fit(inputs, outputs, epochs=3, batch_size=256, callbacks=[early_stopping, lr_reduction, model_checkpoint_acc])

print("Model Saved")