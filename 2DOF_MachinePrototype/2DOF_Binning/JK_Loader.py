import os
import csv
import numpy as np 
import pandas as pd
from tensorflow import keras

def get_data(file_path):
    os.chdir('/home/coder/workspace/Data/2DOF_Machine/Ten_Binning_Data')

    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    PitchRad = data[:, 0]
    YawRad = data[:, 1]  
    Pitch_Dot = data[:, 2]
    Yaw_Dot = data[:, 3]
    DesiredPitch = data[:, 4]
    DesiredYaw = data[:, 5]
    Motor0CurrentRAW = data[:, 6]
    Motor1CurrentRAW = data[:,7]
    TimeRAW = data[:,8]

    Motor0Current = Motor0CurrentRAW / 5000
    Motor1Current = Motor1CurrentRAW / 5000
    Time = TimeRAW / 10000 - 1
    
    return PitchRad, YawRad, Pitch_Dot, Yaw_Dot, DesiredPitch, DesiredYaw, Motor0Current, Motor1Current, Time

def RNN_load_data(file_name, timesteps):
    os.chdir('/home/coder/workspace/Data/2DOF_Machine/Ten_Binning_Data')
    # Load the CSV file
    PitchRad, YawRad, Pitch_Dot, Yaw_Dot, DesiredPitch, DesiredYaw, Motor0Current, Motor1Current, Time = get_data(file_name)

    # spike_limit = 0.15
    # for t in range(len(y)-1):
    #     if abs(y[t]) > abs((y[t+1]+y[t-1])/2) + spike_limit:
    #         y[t] = (y[t+1]+y[t-1])/2
    #     if abs(p[t]) > abs((p[t+1]+p[t-1])/2) + spike_limit:
    #         p[t] = (p[t+1]+p[t-1])/2
    #     if abs(r[t]) > abs((r[t+1]+r[t-1])/2) + spike_limit:
    #         r[t] = (r[t+1]+r[t-1])/2

    # Create data input and output sets
    inputs = []
    for i in range(timesteps, len(PitchRad)):
        timestep_inputs = np.transpose(np.array([PitchRad[i-timesteps:i], YawRad[i-timesteps:i]]))#, 
        # Pitch_Dot[i-timesteps:i], Yaw_Dot[i-timesteps:i], 
        # DesiredPitch[i-timesteps:i], DesiredYaw[i-timesteps:i], 
        # Motor0Current[i-timesteps:i], Motor1Current[i-timesteps:i], 
        #Time[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        
    dataSet = pd.read_csv(file_name).to_numpy() 
    length_data = len(inputs)
    outputData = np.zeros((length_data, 2))
    outputs = np.zeros((0, 2))
    # Check conditions based on the dataset name
    if 'Z' in file_name:
        # Assign [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
        outputData[:, 0] = 1
    # elif 'X' in dataSet:
    #     # Assign [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
    #     outputData[:, 1] = 1
    # elif '3' in dataSet:
    #     # Assign [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
    #     outputData[:, 2] = 1 
    # elif '4' in dataSet:
    #     # Assign [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] to this element of outputData
    #     outputData[:, 3] = 1  
    # elif '5' in dataSet:
    #     # Assign [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] to this element of outputData
    #     outputData[:, 4] = 1
    # elif '6' in dataSet:
    #     # Assign [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] to this element of outputData
    #     outputData[:, 5] = 1
    # elif '7' in dataSet:
    #     # Assign [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] to this element of outputData
    #     outputData[:, 6] = 1
    # elif '8' in dataSet:
    #     # Assign [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] to this element of outputData
    #     outputData[:, 7] = 1
    # elif '9' in dataSet:
    #     # Assign [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] to this element of outputData
    #     outputData[:, 8] = 1
    else:
        # Assign [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] to this element of outputData
        outputData[:, 1] = 1

    print(f'Outputs assigned for {file_name}')
    outputs = np.concatenate((outputs, outputData), axis=0)
    inputs = np.array(inputs)

    return inputs, outputs
