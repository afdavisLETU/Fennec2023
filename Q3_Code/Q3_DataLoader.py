import os
import csv
import random
import numpy as np 
import pandas as pd
from tensorflow import keras

def get_data(file_path):
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    #Read Values from Data Sheets
    xl = pd.ExcelFile(file_path)
    df = xl.parse("IMU_0")
    IMU = np.transpose(np.array([df["GyrX"],df["GyrY"],df["GyrZ"],df["AccX"],df["AccY"],df["AccZ"]],dtype='float'))
    df = xl.parse("RCOU")
    RCOU = np.transpose(np.array([df["C1"],df["C2"],df["C3"],df["C4"],df["C8"]],dtype='float'))
    #Duplicate RCOU Timesteps to match IMU Frequency
    duplicated_array = []
    for row in RCOU:
        duplicated_array.extend([row] * 40) # Number is equal to -> IMU freq. / RCOU freq. = 400Hz / 10Hz = 40
    RCOU = np.array(duplicated_array, dtype='float')
    #Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]
    #Normalize Values
    for col in range(len(RCOU[0])):
        RCOU[:,col] = (RCOU[:,col] - 1000) / 1000
    IMU_scaling = [3,3,3,5,5,5]
    IMU_offsets = [0,0,0,0.5,0,9.81]
    for col in range(len(IMU[0])):
        IMU[:,col] = (IMU[:,col] + IMU_offsets[col]) / IMU_scaling[col]
    print("Data Set Retrieved")
    return IMU, RCOU

def RNN_load_data(file_name, timesteps, data_coeff):
    # Load the CSV file
    IMU, RCOU = csv_get_data(file_name)

    # Create data input and output sets
    inputs = []
    for b in range(timesteps, len(IMU)):
        a = b-timesteps
        timestep_inputs = np.transpose(np.array([IMU[a:b,0],IMU[a:b,1],IMU[a:b,2],IMU[a:b,3],IMU[a:b,4],IMU[a:b,5],RCOU[a:b,0],RCOU[a:b,1],RCOU[a:b,2],RCOU[a:b,3],RCOU[a:b,4]]))
        inputs.append(timestep_inputs)

    num_bins = 3   
    outputData = np.zeros((len(inputs), num_bins))
    outputs = np.zeros((0, num_bins))
    # Check conditions based on the dataset name
    if 'BB' in file_name:
        outputData[:, 1] = 1
    elif 'CC' in file_name:
        outputData[:, 2] = 1
    else:
        outputData[:, 0] = 1

    outputs = np.concatenate((outputs, outputData), axis=0)
    inputs = np.array(inputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    random_indices = sorted(random_indices)
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]
    return inputs, outputs


def csv_get_data(file_path):
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    # Read Values from CSV file
    df = pd.read_csv(file_path)
    IMU = np.transpose(np.array([df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], df.iloc[:,4], df.iloc[:,5]], dtype='float'))
    RCOU = np.transpose(np.array([df.iloc[:,6], df.iloc[:,7], df.iloc[:,8], df.iloc[:,9], df.iloc[:,10]], dtype='float'))
    
    # Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]

    # #Duplicate RCOU Timesteps
    # duplicated_array = []
    # for row in RCOU:
    #     duplicated_array.extend([row] * 40)
    # RCOU = np.array(duplicated_array, dtype='float')
    
    # # Normalize Values
    # for col in range(len(RCOU[0])):
    #     RCOU[:, col] = (RCOU[:, col] - 1000) / 1000
    # IMU_scaling = [3, 3, 3, 5, 5, 5]
    # IMU_offsets = [0, 0, 0, 0.5, 0, 9.81]
    # for col in range(len(IMU[0])):
    #     IMU[:, col] = (IMU[:, col] + IMU_offsets[col]) / IMU_scaling[col]
    
    print("Data Set Retrieved")
    return IMU, RCOU