#Drone Motion Data Loader Code
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import os
import csv
import numpy as np 
import pandas as pd
from tensorflow import keras
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt


def bandPass(order, highStop, lowStop, sampleRate, data):
    '''
    The function uses a Butterworth bandpass filter and returns the filtered data
    
    :param order: order of the filter, int
    :param highStop: high frequency before the roll off starts, int
    :param lowStop: low frequency before the roll off starts, int
    :param sampleRate: sample rate at which the data was collected, int
    :param data: collected data, array_like
    '''
    b, a = butter(order, [highStop,lowStop], 'bandpass', fs=sampleRate)
    return filtfilt(b, a, data)


def get_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    motor_speed = data[:, 1]
    y = data[:, 2]
    p = data[:, 3]
    r = data[:, 4]
    return motor_speed, y, p, r


def RNN_load_data(file_name, timesteps):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    motor_speed, y, p, r = get_data(file_name)
    
    order = 3 # Roll off of 18 dB per octave
    highStop = 0.1 # Hz
    lowStop = 2.5 # Hz
    sampleRate = 25

    motor_speed = bandPass(order, highStop, lowStop, sampleRate, motor_speed)
    y = bandPass(order, highStop, lowStop, sampleRate, y)
    p = bandPass(order, highStop, lowStop, sampleRate, p)
    r = bandPass(order, highStop, lowStop, sampleRate, r)

    # Create data input and output sets
    inputs = []
    outputs = []
    for i in range(timesteps, len(motor_speed)):
        timestep_inputs = np.transpose(np.array([motor_speed[i-timesteps:i], y[i-timesteps:i], p[i-timesteps:i], r[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        outputs.append([y[i], p[i], r[i]])
    
    inputs, outputs = np.array(inputs), np.array(outputs)
    
    return inputs, outputs


def RNN_model_predict(model, readfile, writefile, timesteps, num_predictions):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    motor_speed, y, p, r = get_data(readfile)

    # Number of predictions to make
    num_predictions = num_predictions - timesteps 
    
    # Initial distance
    inputs = np.transpose(np.array([motor_speed[0:timesteps],y[0:timesteps],p[0:timesteps],r[0:timesteps]]))
    actual = np.array([y,p,r])
    predicted = []
        
    # Load the trained model
    model = keras.models.load_model(model)
    
    # Open the CSV file for writing
    with open(writefile, mode='w', newline='') as file:
    
        # Create a writer object for writing rows to the CSV file
        writer = csv.writer(file)
        
        # Write initial distances
        for t in range(timesteps):
            writer.writerow([float(inputs[t,0]),float(inputs[t,1]),float(inputs[t,2])])
            predicted.append([float(inputs[t,0]),float(inputs[t,1]),float(inputs[t,2])])
        
        # Prepare the input data and make predictions
        for i in range(num_predictions):
            # Make the prediction
            print(np.array([inputs]))
            prediction = model.predict(np.array([inputs]))
            print(prediction[0])
            
            # Update the distance array by eliminating the first value and shifting the rest down
            prediction = prediction[0]
            new_input = [motor_speed[i+timesteps], prediction[0],prediction[1],prediction[2]]
            print(new_input)
            inputs = np.append(inputs[1:], new_input)
            #inputs = np.array([inputs])

            # Writes the predicted values to columns after the motor input data
            writer.writerow([prediction[0],prediction[1],prediction[2]])
            predicted.append([prediction[0],prediction[1],prediction[2]])
    return(actual, predicted)
