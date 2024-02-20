import os
import csv
import random
import numpy as np 
import pandas as pd
from tensorflow import keras
from scipy.fft import fft, ifft

def low_pass(data, noise_freq, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_and(freq < noise_freq, freq > -noise_freq)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real

def Micah_Filter(data, window, noise_freq, sampling_rate):
    fft_data = []
    for i in range(len(data)-window+1):
        fft_data.append(low_pass(data[i:window+i], noise_freq, sampling_rate))
    fft_data = np.array(fft_data)

    clean_data = []
    for c in range(1, len(fft_data[0])):
        diagonal = fft_data[:c, c-1::-1]
        clean_data.append(np.mean(diagonal))
    for r in range(len(fft_data) - window + 1):
        diagonal_values = fft_data[r:r + window, window - np.arange(window) - 1]
        clean_data.append(np.mean(diagonal_values))
    x = [np.mean([fft_data[-1 + r - c, -1 - r] for r in range(c + 1)]) for c in range(len(fft_data[0]) - 1)]
    clean_data.extend(reversed(x))

    return np.array(clean_data)

def downsample(data, divisor):
    # Down sampling data through averaging
    rows, cols = len(data), len(data[0])
    new_rows = rows // divisor
    downsampled_data = np.zeros((new_rows, cols))
    for col in range(cols):
        for r in range(new_rows):
            start_row = r * divisor
            end_row = min(start_row + divisor, rows)
            downsampled_data[r, col] = np.mean(data[start_row:end_row, col])
    return downsampled_data

def get_data(file_path):
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    #Read Values from Data Sheets
    xl = pd.ExcelFile(file_path)
    df = xl.parse("IMU_0")
    IMU = np.transpose(np.array([df["GyrX"],df["GyrY"],df["GyrZ"],df["AccX"],df["AccY"],df["AccZ"]],dtype='float'))
    df = xl.parse("RCOU")
    RCOU = np.transpose(np.array([df["C1"],df["C2"],df["C3"],df["C4"],df["C8"]],dtype='float'))
    
    #Filter IMU Data
    window, lp_limit = 100, 10 #lp_limit = low pass limit
    sampling_rate = 400

    stuff = []
    for col in range(len(IMU[0])):
        stuff.append(Micah_Filter(IMU[:,col], window, lp_limit, sampling_rate)) 
    stuff = np.array(stuff)
    IMU = np.transpose(stuff) 
    
    #Duplicate RCOU Timesteps
    duplicated_array = []
    for row in RCOU:
        duplicated_array.extend([row] * 40)
    RCOU = np.array(duplicated_array, dtype='float')

    #Data Length Correction
    RCOU = RCOU[:len(IMU)]
    IMU = IMU[:len(RCOU)]

    #Normalize Values
    RCOU_scaling = [1300,1400,1500,1500,1100]
    RCOU_offsets = [200,200,200,100,500]
    for col in range(len(RCOU[0])):
        RCOU[:,col] = (RCOU[:,col] - RCOU_offsets[col]) / RCOU_scaling[col]
    IMU_scaling = [.3,.3,.3,.5,.5,.5]
    IMU_offsets = [0,0,0,0.5,0,9.81]
    for col in range(len(IMU[0])):
        IMU[:,col] = (IMU[:,col] + IMU_offsets[col]) / IMU_scaling[col]

    return IMU, RCOU

def csv_load_data(file_name, timesteps, data_coeff, output):
    # Load the CSV file
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    df = pd.read_csv(file_name)
    data = np.array(df.values)
    # Create data input and output sets
    inputs = []
    outputs = []
    for b in range(timesteps,len(data)):
        a = b-timesteps
        inputs.append(data[a:b])
        outputs.append(data[b,output])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    random_indices = sorted(random_indices)
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]

    return inputs, outputs


def csv_model_predict(model_sim, test_data, timesteps, output, num_predictions, pred_offset):
    # Load the CSV file
    os.chdir('/home/coder/workspace/Data/Becky_Data/')
    df = pd.read_csv(test_data)
    data = np.array(df.values)

    # Create data input and output sets
    a, b = pred_offset, (pred_offset + timesteps)
    inputs = np.array(data[a:b])
    actual = np.array(data[a:a+num_predictions,output])
    predictions = np.array(data[a:b,output])

    # Load the trained model
    model = keras.models.load_model(model_sim)
    for i in range(num_predictions - timesteps):
        print(i+timesteps, "of", num_predictions, end='\r')
        predicted = np.array(model.predict(np.array([inputs]),verbose=0))
        # Update the input data by eliminating the first timestep and adding the prediction
        c = i + timesteps + pred_offset
        new_input = data[c]
        new_input[output]=predicted[0]
        predictions = np.append(predictions, predicted[0])
        inputs = np.concatenate([inputs[1:], [new_input]])

    return actual, predictions

def iter_gen(model_name, inputs, outputs, timesteps, output, pred_prob):
    #Iterative Generalization

    # Load the trained model
    model = keras.models.load_model(model_name)

    for i in range(len(data) - timesteps):
        inputs.append(input_data)
        c = i + timesteps
        new_input = data[c]
        print(i+timesteps, "of", len(data), end='\r')
        if random.uniform(0, 100) < pred_prob:
            predicted = np.array(model.predict(np.array([input_data]),verbose=0))
            new_input[output]=predicted[0]
        input_data = np.concatenate([input_data[1:], [new_input]])
    inputs = np.array(inputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    random_indices = sorted(random_indices)
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]

    return inputs, outputs

def excel_load_data(file_name, timesteps, data_coeff, output):
    # Load the CSV file
    IMU, RCOU = get_data(file_name)

    #Downsample
    IMU_50Hz = downsample(IMU, 8)
    RCOU_50Hz = downsample(RCOU, 8)

    # Create data input and output sets
    inputs = []
    outputs = []
    for b in range(timesteps, len(IMU_50Hz)):
        a = b-timesteps
        timestep_inputs = np.transpose(np.array([IMU_50Hz[a:b,0],IMU_50Hz[a:b,1],IMU_50Hz[a:b,2],IMU_50Hz[a:b,3],IMU_50Hz[a:b,4],IMU_50Hz[a:b,5],RCOU_50Hz[a:b,0],RCOU_50Hz[a:b,1],RCOU_50Hz[a:b,2],RCOU_50Hz[a:b,3],RCOU_50Hz[a:b,4]]))
        inputs.append(timestep_inputs)

        timestep_outputs = np.transpose(np.array([IMU_50Hz[b,output]]))#,IMU_50Hz[a:b,1],IMU_50Hz[a:b,2]]))
        outputs.append(timestep_outputs)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    #Random Data Sampling
    random_indices = random.sample(range(len(inputs)), int(data_coeff*len(inputs)))
    random_indices = sorted(random_indices)
    inputs = [inputs[i] for i in random_indices]
    outputs = [outputs[i] for i in random_indices]

    return inputs, outputs


def excel_model_predict(model_sim, test_data, timesteps, output, num_predictions, pred_offset):
    # Load the CSV file
    IMU, RCOU = get_data(test_data)

    #Downsample
    IMU_50Hz = downsample(IMU, 8)
    RCOU_50Hz = downsample(RCOU, 8)

    # Initialize Inputs, Predicted, and Actual Arrays
    a, b = pred_offset, (pred_offset + timesteps)
    inputs = np.transpose(np.array([IMU_50Hz[a:b,0],IMU_50Hz[a:b,1],IMU_50Hz[a:b,2],IMU_50Hz[a:b,3],IMU_50Hz[a:b,4],IMU_50Hz[a:b,5],RCOU_50Hz[a:b,0],RCOU_50Hz[a:b,1],RCOU_50Hz[a:b,2],RCOU_50Hz[a:b,3],RCOU_50Hz[a:b,4]]))
    predictions = np.transpose(np.array([IMU_50Hz[a:b, output]]))
    actual = np.transpose(np.array([IMU_50Hz[a:, output]]))

    # Load the trained model
    model = keras.models.load_model(model_sim)
    
    # Open the CSV file for writing
    for i in range(num_predictions - timesteps):
        print(i+timesteps, "of", num_predictions, end='\r')
        predicted = np.array([model.predict(np.array([inputs]),verbose=0)])

        # Update the input data by eliminating the first timestep and adding the prediction
        c = i + timesteps + pred_offset
        new_input = np.transpose(np.array([IMU_50Hz[c,0],IMU_50Hz[c,1],IMU_50Hz[c,2],IMU_50Hz[c,3],IMU_50Hz[c,4],IMU_50Hz[c,5],RCOU_50Hz[c,0],RCOU_50Hz[c,1],RCOU_50Hz[c,2],RCOU_50Hz[c,3],RCOU_50Hz[c,4]]))
        new_input[output]=predicted[0]
        predictions = np.concatenate([predictions, predicted[0]])
        inputs = np.concatenate([inputs[1:], [new_input]])

    return actual[:len(predictions)], predictions