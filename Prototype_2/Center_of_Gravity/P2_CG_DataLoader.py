import os
import csv
import numpy as np 
import pandas as pd
from tensorflow import keras

def get_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    motor_speed = data[:, 0]
    y = data[:, 1]
    p = data[:, 2]
    r = data[:, 3]
    return motor_speed, y, p, r

def RNN_load_data(file_name, timesteps):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    motor_speed, y, p, r = get_data(file_name)

    spike_limit = 0.15
    for t in range(len(y)-1):
        if abs(y[t]) > abs((y[t+1]+y[t-1])/2) + spike_limit:
            y[t] = (y[t+1]+y[t-1])/2
        if abs(p[t]) > abs((p[t+1]+p[t-1])/2) + spike_limit:
            p[t] = (p[t+1]+p[t-1])/2
        if abs(r[t]) > abs((r[t+1]+r[t-1])/2) + spike_limit:
            r[t] = (r[t+1]+r[t-1])/2

    # Create data input and output sets
    inputs = []
    for i in range(timesteps, len(motor_speed)):
        timestep_inputs = np.transpose(np.array([motor_speed[i-timesteps:i], y[i-timesteps:i], p[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        
    dataset = pd.read_csv(file_name).to_numpy()
    length_data = len(inputs)
    outputData = np.zeros((length_data, 3))
    outputs = np.zeros((0, 3))
    # Check conditions based on the dataset name
    if 'rw' in file_name:
        # Assign [0, 1, 0] to the second element of outputData
        outputData[:, 1] = 1
    elif 'lw' in file_name:
        # Assign [0, 0, 1] to the first element of outputData
        outputData[:, 2] = 1
    else:
        # Assign [1, 0, 0] to the third element of outputData
        outputData[:, 0] = 1

    outputs = np.concatenate((outputs, outputData), axis=0)
    inputs = np.array(inputs)

    #Troubleshooting
    print(outputs)
    # print(inputs.shape)

    return inputs, outputs

def RNN_model_predict(model_1, model_2, readfile, writefile, timesteps, num_predictions):
    os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')
    # Load the CSV file
    motor_speed, y, p, r = get_data(readfile)

    spike_limit = 0.15
    for t in range(len(y)-1):
        if abs(y[t]) > abs((y[t+1]+y[t-1])/2) + spike_limit:
            y[t] = (y[t+1]+y[t-1])/2
        if abs(p[t]) > abs((p[t+1]+p[t-1])/2) + spike_limit:
            p[t] = (p[t+1]+p[t-1])/2
        if abs(r[t]) > abs((r[t+1]+r[t-1])/2) + spike_limit:
            r[t] = (r[t+1]+r[t-1])/2

    # Number of predictions to make
    num_predictions = num_predictions - timesteps 

    # Initial distance
    inputs = np.transpose(np.array([motor_speed[0:timesteps],y[0:timesteps],p[0:timesteps]]))
    actual = np.transpose(np.array([y,p]))
    predicted = []
        
    # Load the trained model
    model_y = keras.models.load_model(model_1)
    model_p = keras.models.load_model(model_2)

    # Open the CSV file for writing
    with open(writefile, mode='w', newline='') as file:
    
        # Create a writer object for writing rows to the CSV file
        writer = csv.writer(file)
        
        # Write initial distances
        for t in range(timesteps):
            writer.writerow([float(inputs[t,1]),float(inputs[t,2]),y[t],p[t]])
            predicted.append([float(inputs[t,1]),float(inputs[t,2])])

        # Prepare the input data and make predictions
        for i in range(num_predictions):
            # Make the prediction
            prediction = np.array([float(model_y.predict(np.array([inputs]))),float(model_p.predict(np.array([inputs])))])

            # Update the distance array by eliminating the first value and shifting the rest down
            new_input = [motor_speed[i+timesteps], prediction[0],prediction[1]]
            #new_input = [motor_speed[i+timesteps], y[i+timesteps],p[i+timesteps]]
            inputs = np.concatenate([inputs[1:], [new_input]])

            # Writes the predicted values to columns after the motor input data
            writer.writerow([prediction[0],prediction[1],y[i+timesteps],p[i+timesteps]])
            predicted.append([prediction[0],prediction[1]])
    return np.array(actual), np.array(predicted)
