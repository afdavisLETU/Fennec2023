import os
import csv
import random
import numpy as np 
import pandas as pd
from tensorflow import keras
from scipy.interpolate import interp1d

def get_data(file_path):
    os.chdir('/home/coder/workspace/Data/Q3_Data/')
    #Read Values from Data Sheets
    xl = pd.ExcelFile(file_path)
    df = xl.parse("IMU_0")
    IMU = np.transpose(np.array([df["GyrX"],df["GyrY"],df["GyrX"],df["AccX"],df["AccY"],df["AccZ"]],dtype='float'))
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


def moving_avg_and_downsample(data, window_size, original_sampling_rate, target_sampling_rate, axis=0):
    # Compute the moving average along the specified axis
    moving_avg = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='valid'), axis, data)
    
    # Compute the decimation factor
    decimation_factor = original_sampling_rate // target_sampling_rate
    
    # Downsample the moving average array
    downsampled_avg = np.take(moving_avg, np.arange(0, moving_avg.shape[axis], decimation_factor), axis)
    
    return downsampled_avg


def upsample_variable(data, original_sample_rate, target_sample_rate):
    # Generate timestamps for the original data
    original_timestamps = np.linspace(0, (data.shape[0] - 1) / original_sample_rate, data.shape[0])
    
    # Generate timestamps for the target data
    target_timestamps = np.arange(0, original_timestamps[-1], 1 / target_sample_rate)
    
    # Initialize the upsampled data array
    upsampled_data = np.empty((len(target_timestamps),) + data.shape[1:])
    
    # Interpolate along each dimension separately
    for i in range(data.shape[1]):
        interpolator = interp1d(original_timestamps, data[:, i], kind='linear', fill_value='extrapolate')
        upsampled_data[:, i] = interpolator(target_timestamps)
    
    return upsampled_data, target_timestamps


def RNN_load_data(file_name, timesteps, data_coeff):
    # Load the CSV file
    IMU_400Hz, RCOU_10Hz = get_data(file_name)

    window_size = 8
    IMU_Original_Sample_Rate = 400 #Hz
    RCOU_Original_Sample_Rate = 10 #Hz
    Desired_Sample_Rate = 50 #Hz

    IMU_50Hz = moving_avg_and_downsample(IMU_400Hz, window_size, IMU_Original_Sample_Rate, Desired_Sample_Rate, axis=0)
    RCOU_50Hz, _ = upsample_variable(RCOU_10Hz, RCOU_Original_Sample_Rate, Desired_Sample_Rate)

    # Create data input and output sets
    inputs = []
    outputs = []
    for b in range(timesteps, len(IMU_50Hz)):
        a = b-timesteps
        timestep_inputs = np.transpose(np.array([IMU_50Hz[a:b,0],IMU_50Hz[a:b,1],IMU_50Hz[a:b,2],RCOU_50Hz[a:b,0],RCOU_50Hz[a:b,1],RCOU_50Hz[a:b,2],RCOU_50Hz[a:b,3]]))
        inputs.append(timestep_inputs)

        timestep_outputs = np.transpose(np.array([IMU_50Hz[a:b,0]]))#,IMU_50Hz[a:b,1],IMU_50Hz[a:b,2]]))
        outputs.append(timestep_outputs)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]
    return inputs, outputs