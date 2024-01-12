#Drone Motion Data Loader Code
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import csv
import numpy as np 
import pandas as pd
from tensorflow import keras

def high_pass(data, noise_frequency, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_or(freq > noise_frequency, freq < -noise_frequency)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real  # Extract real values

def low_pass(data, noise_frequency, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_and(freq < noise_frequency, freq > -noise_frequency)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real  # Extract real values

def get_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    motor_speed = data[:, 1]
    y = data[:, 2]
    p = data[:, 3]
    r = data[:, 4]
    return motor_speed, y, p, r

def RNN_load_data(file_name, timesteps):
    # Load the CSV file
    motor_speed, y, p, r = get_data(file_name)
    
    # Create data input and output sets
    inputs = []
    outputs = []
    for i in range(timesteps, len(data)):
        timestep_inputs = np.transpose(np.array([motor_speed[i-timesteps:i], y[i-timesteps:i], p[i-timesteps:i], r[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        outputs.append([y[i], p[i], r[i]])
    
    inputs, outputs = np.array(inputs), np.array(outputs)
    
    return inputs, outputs

def RNN_model_predict(model, readfile, writefile, timesteps, num_predictions):
    # Load the CSV file
    data = pd.read_csv(readfile)
    
    # Number of predictions to make
    num_predictions = num_predictions - timesteps 
    
    # Initial distance
    distance = data.iloc[0:timesteps, 1].values
    actual = data.iloc[:,1].values
    predicted = []
        
    # Load the trained model
    model = keras.models.load_model(model)
    
    # Open the CSV file for writing
    with open(writefile, mode='w', newline='') as file:
    
        # Create a writer object for writing rows to the CSV file
        writer = csv.writer(file)
        
        # Write initial distances
        for t in range(timesteps):
            writer.writerow([float(distance[t]*35)])
            predicted.append(float(distance[t]*35))
        
        # Prepare the input data and make predictions
        for i in range(num_predictions):
            # Load input data
            motor_speed = data.iloc[i:i+timesteps, 0].values
            inputs = np.transpose(np.array([motor_speed, distance])).reshape(1,timesteps,2)
            
            # Make the prediction
            prediction = model.predict(inputs)
            print(prediction*35)
            
            # Update the distance array by eliminating the first value and shifting the rest down
            distance = np.append(distance[1:], prediction)
            
            # Writes the predicted values to columns after the motor input data
            writer.writerow([float(prediction*35)])
            predicted.append(float(prediction*35))
    return(actual, predicted)
