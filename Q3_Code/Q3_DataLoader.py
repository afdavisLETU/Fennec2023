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
    RCOU = np.transpose(np.array([df["C1"],df["C2"],df["C3"],df["C4"]],dtype='float'))
    df = xl.parse("ATT")
    ATT = np.transpose(np.array([df["DesPitch"],df["Pitch"]],dtype='float'))
    #Duplicate RCOU Timesteps to match IMU Frequency
    duplicated_array = []
    for row in RCOU:
        duplicated_array.extend([row] * 40) # Number is equal to -> IMU freq. / RCOU freq. = 400Hz / 10Hz = 40
    RCOU = np.array(duplicated_array, dtype='float')
    #Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]
    ATT = ATT[:len(IMU)]
    if len(ATT) != len(IMU):
        print('Length Mismatch')
    #Normalize Values
    for col in range(len(RCOU[0])):
        RCOU[:,col] = (RCOU[:,col] - 1000) / 1000
    IMU_scaling = [3,3,3,5,5,5]
    IMU_offsets = [0,0,0,0.5,0,9.81]
    for col in range(len(IMU[0])):
        IMU[:,col] = (IMU[:,col] + IMU_offsets[col]) / IMU_scaling[col]
    ATT_scaling = [13,11]
    for col in range(len(ATT[0])):
        ATT[:,col] = (ATT[:,col] / ATT_scaling[col])
    print("Data Set Retrieved")
    return IMU, RCOU, ATT

def RNN_load_data(file_name, timesteps, data_coeff):
    # Load the CSV file
    IMU, RCOU, ATT = csv_get_data(file_name)

    # Create data input and output sets
    inputs = []
    for b in range(timesteps, len(IMU)):
        a = b-timesteps
        timestep_inputs = np.transpose(np.array([IMU[a:b,0],IMU[a:b,1],IMU[a:b,2],IMU[a:b,3],IMU[a:b,4],IMU[a:b,5],RCOU[a:b,0],RCOU[a:b,1],RCOU[a:b,2],RCOU[a:b,3],ATT[a:b,0],ATT[a:b,1]]))
        inputs.append(timestep_inputs)

    num_bins = 5   
    outputData = np.zeros((len(inputs), num_bins))
    outputs = np.zeros((0, num_bins))
    # Check conditions based on the dataset name
    if 'BB' in file_name:
        outputData[:, 1] = 1
    elif 'CC' in file_name:
        outputData[:, 2] = 1
    elif 'DD' in file_name:
        outputData[:,3] = 1
    elif 'EE' in file_name:
        outputData[:,4] = 1
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
    RCOU = np.transpose(np.array([df.iloc[:,6], df.iloc[:,7], df.iloc[:,8], df.iloc[:,9]], dtype='float'))
    ATT = np.transpose(np.array([df.iloc[:,10], df.iloc[:,11]], dtype='float'))
    
    # Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]
    ATT = ATT[:len(IMU)]
    
    print("Data Set Retrieved")
    return IMU, RCOU, ATT


def trim_data_to_min_file_length(data_categories):
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    # Load the data from the files in each category
    data = [[pd.read_csv(file) for file in category] for category in data_categories]

    # Calculate the length of each file
    file_lengths = [[len(df) for df in category] for category in data]

    # Flatten the file lengths
    flat_file_lengths = [length for sublist in file_lengths for length in sublist]

    # Find the minimum file length
    min_length = min(flat_file_lengths)

    # Trim the data length of each file to the minimum length
    trimmed_data = [[df[:min_length] for df in category] for category in data]

    # Save the trimmed data as CSV files with "TrimmedTemp_" prepended to the original file name
    trimmed_file_names = []
    for category_data, category_files in zip(trimmed_data, data_categories):
        category_file_names = []
        for df, file in zip(category_data, category_files):
            trimmed_file_name = f"TempTrimmed_{file}"
            df.to_csv(trimmed_file_name, index=False)
            category_file_names.append(trimmed_file_name)
        trimmed_file_names.append(category_file_names)

    return trimmed_file_names