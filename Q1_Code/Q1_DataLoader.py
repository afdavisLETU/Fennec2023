import os
import csv
import random
import numpy as np 
import pandas as pd
from tensorflow import keras
from scipy.interpolate import interp1d

def get_data(file_path):
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    #Read Values from Data Sheets
    xl = pd.ExcelFile(file_path)
    df = xl.parse("IMU_0")
    IMU = np.transpose(np.array([df["GyrX"],df["GyrY"],df["GyrX"],df["AccX"],df["AccY"],df["AccZ"]],dtype='float'))
    df = xl.parse("RCOU")
    RCOU = np.transpose(np.array([df["C1"],df["C2"],df["C3"],df["C4"],df["C8"]],dtype='float'))
    #Duplicate RCOU Timesteps to 50Hz
    duplicated_array = []
    for row in RCOU:
        duplicated_array.extend([row] * 5)
    RCOU_50Hz = np.array(duplicated_array, dtype='float')
    #Downsample IMU to 50Hz
    cols, rows, new_rows = len(IMU[0]), len(IMU), len(IMU) // 8
    IMU_50Hz = np.zeros((new_rows, cols))
    for col in range(cols):
        for i in range(new_rows):
            start_row = i * 8
            end_row = min(start_row + 8, rows)
            IMU_50Hz[i, col] = np.mean(IMU[start_row:end_row, col])
    #Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]
    #Normalize Values
    for col in range(len(RCOU[0])):
        RCOU[:,col] = (RCOU[:,col] - 1000) / 1000
    IMU_scaling = [.3,.3,.3,.5,.5,.5]
    IMU_offsets = [0,0,0,0.5,0,9.81]
    for col in range(len(IMU[0])):
        IMU[:,col] = (IMU[:,col] + IMU_offsets[col]) / IMU_scaling[col]
    return IMU, RCOU

def RNN_load_data(file_name, timesteps, data_coeff):
    # Load the CSV file
    IMU_50Hz, RCOU_50Hz = get_data(file_name)

    # Create data input and output sets
    inputs = []
    outputs = []
    for b in range(timesteps, len(IMU_50Hz)):
        a = b-timesteps
        timestep_inputs = np.transpose(np.array([IMU_50Hz[a:b,0],IMU_50Hz[a:b,1],IMU_50Hz[a:b,2],IMU_50Hz[a:b,3],IMU_50Hz[a:b,4],IMU_50Hz[a:b,5],RCOU_50Hz[a:b,0],RCOU_50Hz[a:b,1],RCOU_50Hz[a:b,2],RCOU_50Hz[a:b,3],RCOU_50Hz[a:b,4]]))
        inputs.append(timestep_inputs)

        timestep_outputs = np.transpose(np.array([IMU_50Hz[b,5]]))#,IMU_50Hz[a:b,1],IMU_50Hz[a:b,2]]))
        outputs.append(timestep_outputs)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    random_indices = sorted(random_indices)
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]

    return inputs, outputs


def RNN_model_predict(model_sim, test_data, timesteps, num_predictions, pred_offset):
    # Load the data
    IMU_50Hz, RCOU_50Hz = get_data(test_data)

    # Initialize Inputs, Predicted, and Actual Arrays
    a, b = pred_offset, (pred_offset + timesteps)
    inputs = np.transpose(np.array([IMU_50Hz[a:b,0],IMU_50Hz[a:b,1],IMU_50Hz[a:b,2],IMU_50Hz[a:b,3],IMU_50Hz[a:b,4],IMU_50Hz[a:b,5],RCOU_50Hz[a:b,0],RCOU_50Hz[a:b,1],RCOU_50Hz[a:b,2],RCOU_50Hz[a:b,3],RCOU_50Hz[a:b,4]]))
    predictions = np.transpose(np.array([IMU_50Hz[a:b,5]]))
    actual = np.transpose(np.array([IMU_50Hz[a:,5]]))

    # Load the trained model
    model = keras.models.load_model(model_sim)
    
    # Open the CSV file for writing
    for i in range(num_predictions - timesteps):
        print(i+timesteps, "of", num_predictions, end='\r')
        predicted = np.array([model.predict(np.array([inputs]),verbose=0)])

        # Update the input data by eliminating the first timestep and adding the prediction
        a = i + timesteps
        new_input = np.transpose(np.array([IMU_50Hz[a,0],IMU_50Hz[a,1],IMU_50Hz[a,2],IMU_50Hz[a,3],IMU_50Hz[a,4],IMU_50Hz[a,5],RCOU_50Hz[a,0],RCOU_50Hz[a,1],RCOU_50Hz[a,2],RCOU_50Hz[a,3],RCOU_50Hz[a,4]]))
        new_input[5]=predicted[0]
        predictions = np.concatenate([predictions, predicted[0]])
        inputs = np.concatenate([inputs[1:], [new_input]])

    return actual[:len(predictions)], predictions