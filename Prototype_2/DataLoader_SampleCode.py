#Drone Motion Data Loader Code
#Written By: Micah Heikkila
#Last Modified: July 2,2023

import csv
import numpy as np 
import pandas as pd
from tensorflow import keras

def FNN_load_data(file_name, timesteps):
    # Load the CSV file
    data = pd.read_csv(file_name)
    
    # Read Data
    motor_speed = data.iloc[:, 0].values  
    distance = data.iloc[:, 1].values
    
    # Normalize data
    motor_speed = motor_speed / 1000
    distance = distance / 35
    
    # Create data input and output sets
    inputs = []
    outputs = []
    for i in range(timesteps, len(data)):
        timestep_inputs = np.concatenate((motor_speed[i-timesteps:i], distance[i-timesteps:i]), axis=0)
        inputs.append(timestep_inputs)
        outputs.append(distance[i])
    
    inputs, outputs = np.array(inputs), np.array(outputs)
    
    return inputs, outputs

def RNN_load_data(file_name, timesteps):
    # Load the CSV file
    data = pd.read_csv(file_name)
    
    # Read Data
    motor_speed = data.iloc[:, 0].values  
    distance = data.iloc[:, 1].values
    
    # Normalize data
    motor_speed = motor_speed / 1000
    distance = distance / 35
    
    # Create data input and output sets
    inputs = []
    outputs = []
    for i in range(timesteps, len(data)):
        timestep_inputs = np.transpose(np.array([motor_speed[i-timesteps:i], distance[i-timesteps:i]]))
        inputs.append(timestep_inputs)
        outputs.append(distance[i])
    
    inputs, outputs = np.array(inputs), np.array(outputs)
    
    return inputs, outputs

def FNN_model_predict(model, readfile, writefile, timesteps, num_predictions):
    # Load the CSV file
    data = pd.read_csv(readfile)
    
    # Number of predictions to make
    num_predictions = num_predictions - timesteps 
    
    # Initial distance
    distance = data.iloc[0:timesteps, 1].values
    distance = distance / 35
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
            writer.writerow([float(distance[t])])
            predicted.append(float(distance[t]*35))
        
        # Prepare the input data and make predictions
        for i in range(num_predictions):
            # Load input data
            motor_speed = data.iloc[i:i+timesteps, 0].values
            motor_speed = motor_speed / 1000
            inputs = np.array([motor_speed, distance]).reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(inputs)
            print(prediction*35)
            
            # Update the distance array by eliminating the first value and shifting the rest down
            distance = np.append(distance[1:], prediction)
            
            # Writes the predicted values to columns after the motor input data
            writer.writerow([float(prediction*35)])
            predicted.append(float(prediction*35))
    return(actual, predicted)

def RNN_model_predict(model, readfile, writefile, timesteps, num_predictions):
    # Load the CSV file
    data = pd.read_csv(readfile)
    
    # Number of predictions to make
    num_predictions = num_predictions - timesteps 
    
    # Initial distance
    distance = data.iloc[0:timesteps, 1].values
    distance = distance / 35
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
            motor_speed = motor_speed / 1000
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
