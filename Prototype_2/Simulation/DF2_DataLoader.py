#Drone Motion Data Loader Code
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import os
import csv
import numpy as np 
import pandas as pd
from tensorflow import keras

def get_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    motor_speed = data[:, 0]
    p = data[:, 0]
    y = data[:, 1]
    y = y / 10
    m0 = data[:, 6]
    m1 = data[:, 7]
    t_err = data[:,8]

    spike_limit = 5
    for t in range(len(p)-1):
        if abs(p[t]) > abs((p[t+1]+p[t-1])/2) + spike_limit:
            p[t] = (p[t+1]+p[t-1])/2
        if abs(y[t]) > abs((y[t+1]+y[t-1])/2) + spike_limit:
            y[t] = (y[t+1]+y[t-1])/2
    return p, y, m0, m1, t_err

def RNN_load_data(file_name, timesteps):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    p, y, m0, m1, t_err = get_data(file_name)

    # Create data input and output sets
    inputs = []
    outputs = []
    for i in range(timesteps, len(m0)):
        timestep_inputs = np.transpose(np.array([p[i-timesteps:i], y[i-timesteps:i], m0[i-timesteps:i], m1[i-timesteps:i], t_err[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        outputs.append([p[i], y[i]])
    
    inputs, outputs = np.array(inputs), np.array(outputs)
    
    return inputs, outputs

def RNN_model_predict(model_1, model_2, readfile, writefile, timesteps, num_predictions):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    p, y, m0, m1, t_err = get_data(readfile)

    # Number of predictions to make
    num_predictions = num_predictions - timesteps 

    # Initial distance
    inputs = np.transpose(np.array([p[0:timesteps],y[0:timesteps],m0[0:timesteps],m1[0:timesteps],t_err[0:timesteps]]))
    actual = np.transpose(np.array([p,y]))
    predicted = []
    # Load the trained model
    model_p = keras.models.load_model(model_1)
    model_y = keras.models.load_model(model_2)

    # Open the CSV file for writing
    with open(writefile, mode='w', newline='') as file:
    
        # Create a writer object for writing rows to the CSV file
        writer = csv.writer(file)
        
        # Write initial distances
        for t in range(timesteps):
            writer.writerow([float(inputs[t,0]),float(inputs[t,1]),p[t],y[t]])
            predicted.append([float(inputs[t,0]),float(inputs[t,1])])

        # Prepare the input data and make predictions
        for i in range(num_predictions):
            # Make the prediction
            prediction = np.array([float(model_p.predict(np.array([inputs]))),float(model_y.predict(np.array([inputs])))])

            # Update the distance array by eliminating the first value and shifting the rest down
            new_input = [prediction[0],prediction[1],m0[i+timesteps],m1[i+timesteps],t_err[i+timesteps]]
            #new_input = [motor_speed[i+timesteps], y[i+timesteps],p[i+timesteps]]
            inputs = np.concatenate([inputs[1:], [new_input]])

            # Writes the predicted values to columns after the motor input data
            writer.writerow([prediction[0],prediction[1],p[i+timesteps],y[i+timesteps]])
            predicted.append([prediction[0],prediction[1]])
    return np.array(actual), np.array(predicted)
