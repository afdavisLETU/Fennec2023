import csv
import random
import numpy as np 
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from scipy.fft import fft, ifft

def low_pass(data, noise_freq, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_and(freq < noise_freq, freq > -noise_freq)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real

def Micah_Filter(IMU, window, noise_freq, sampling_rate):
    clean_IMU = []
    for col in range(len(IMU[0])):
        data = IMU[:,col]
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
        clean_IMU.append(np.array(clean_data))
    clean_IMU = np.transpose(np.array(clean_IMU)) 

    return clean_IMU

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
    #Read Values from Data Sheets
    xl = pd.ExcelFile(file_path)
    df = xl.parse("BARO")
    BARO = np.transpose(np.array([df["Alt"]],dtype='float'))
    df = xl.parse("ATT")
    ATT = np.transpose(np.array([df["Roll"],df["Pitch"]],dtype='float'))
    df = xl.parse("GPS")
    GPS = np.transpose(np.array([df["Spd"]],dtype='float'))
    df = xl.parse("IMU_0")
    IMU = np.transpose(np.array([df["GyrX"],df["GyrY"],df["GyrZ"],df["AccX"],df["AccY"],df["AccZ"]],dtype='float'))
    df = xl.parse("RCOU")
    RCOU = np.transpose(np.array([df["C1"],df["C2"],df["C3"],df["C4"],df["C8"]],dtype='float'))
    
    #Upsample BARO, ATT, and RCOU Timesteps
    duplicated_array = []
    for row in BARO:
        duplicated_array.extend([row] * 20)
    BARO = np.array(duplicated_array, dtype='float')

    duplicated_array = []
    for row in GPS:
        duplicated_array.extend([row] * 80)
    GPS = np.array(duplicated_array, dtype='float')

    duplicated_array = []
    for row in RCOU:
        duplicated_array.extend([row] * 40)
    RCOU = np.array(duplicated_array, dtype='float')

    #Data Length Correction
    trim = np.min([len(BARO),len(ATT),len(IMU),len(RCOU),len(GPS)])
    print("Data Length:", trim)
    RCOU = RCOU[:trim]
    IMU = IMU[:trim]
    BARO = BARO[:trim]
    ATT = ATT[:trim]
    GPS = GPS[:trim]

    #Normalize Values
    GPS_scaling = [20]
    GPS_offsets = [0]
    BARO_scaling = [50]
    BARO_offsets = [0]
    ATT_scaling = [50,50]
    ATT_offsets = [0,0]
    RCOU_scaling = [100,100,100,100,1000]
    RCOU_offsets = [1650,1650,1650,1500,1000]
    IMU_scaling = [1,1,1,1,2,5]
    IMU_offsets = [0,0,0,0,.75,9.81]
    for col in range(len(GPS[0])):
        GPS[:,col] = (GPS[:,col] - GPS_offsets[col]) / GPS_scaling[col]
    for col in range(len(BARO[0])):
        BARO[:,col] = (BARO[:,col] - BARO_offsets[col]) / BARO_scaling[col]
    for col in range(len(RCOU[0])):
        RCOU[:,col] = (RCOU[:,col] - RCOU_offsets[col]) / RCOU_scaling[col]
    for col in range(len(IMU[0])):
        IMU[:,col] = (IMU[:,col] + IMU_offsets[col]) / IMU_scaling[col]
    for col in range(len(ATT[0])):
        ATT[:,col] = (ATT[:,col] + ATT_offsets[col]) / ATT_scaling[col]

    return BARO, ATT, GPS, IMU, RCOU

def load_data(file_name, timesteps, data_coeff, output):
    # Load the CSV file
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


def model_predict(model_name, test_data, timesteps, output, num_predictions, pred_offset):
    # Load the CSV file
    df = pd.read_csv(test_data)
    data = np.array(df.values)

    # Create data input and output sets
    a, b = pred_offset, (pred_offset + timesteps)
    inputs = np.array(data[a:b])
    actual = np.array(data[a:a+num_predictions,output])
    predictions = np.array(data[a:b,output])

    # Load the trained model
    model = keras.models.load_model(model_name)
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

def model_simulation(test_data, timesteps, num_predictions, pred_offset):
    # Load the CSV file
    df = pd.read_csv(test_data)
    data = np.array(df.values)

    # Create data input and output sets
    a, b = pred_offset, (pred_offset + timesteps)
    inputs = np.array(data[a:b])
    actual = np.array(data[a:a+num_predictions,:10])
    predictions = np.array(data[a:b,:10])
    # Load the trained model
    models = []
    for i in range(10):
        model = keras.models.load_model(f"Model_{i}.h5")
        models.append(model)

    for i in range(num_predictions - timesteps):
        print(i+timesteps, "of", num_predictions, end='\r')
        predicted = []
        for model in models:
            prediction = np.array(model.predict(np.array([inputs]),verbose=0))
            predicted.append(prediction[0,0])
        # Update the input data by eliminating the first timestep and adding the prediction
        c = i + timesteps + pred_offset
        new_input = data[c]
        for i in range(10):
            new_input[i]=predicted[i]
        predictions = np.append(predictions, np.array([predicted]), axis=0)
        inputs = np.concatenate([inputs[1:], [new_input]])

    return actual, predictions



