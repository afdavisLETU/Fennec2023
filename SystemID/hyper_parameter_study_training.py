import csv
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import os
from tensorflow.keras import backend as K
import h5py



# Define hyperparameter samples
training_rate = np.logspace(-5,-1,10) # between 10^-5 to 10^-1
epochs = np.linspace(500,5000,10, dtype=int)

for i, rate in enumerate(training_rate):
    for ii, epoch in enumerate(epochs):

        K.clear_session()

        os.chdir('/home/coder/workspace/VanderWiel/Fennec2023/SystemID')
        name = 'simulated_outputs.csv'
        data = pd.read_csv(name)

        # resetting RAM in gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


        def normalize_data(data):
            """
            Normalize data between 0 and 1 using Min-Max scaling for pandas DataFrame.
            
            Parameters:
            data (pandas DataFrame): Input data to be normalized.
            
            Returns:
            pandas DataFrame: Normalized data.
            """
            min_vals = data.min()
            max_vals = data.max()
            normalized_data = (data - min_vals) / (max_vals - min_vals)
            return normalized_data


        def lowpass(order, highStop, sampleRate, data):
            '''
            The function uses a Butterworth low pass filter and returns the filtered time-series data

            :param order: order of the filter (higher order means faster roll off), int
            :param highStop: high frequency before the roll off starts, int
            :param sampleRate: sample rate at which the data was collected, int
            :param data: collected data, array_like
            '''
            b, a = butter(order, highStop, 'lowpass', fs=sampleRate)
            return filtfilt(b, a, data)


        t = data.iloc[1:, 0].values
        Vp = data.iloc[1:, 1].values
        Vy = data.iloc[1:, 2].values
        pitch = data.iloc[1:, 3].values
        yaw = data.iloc[1:, 4].values

        # Compute velocity derivatives
        tstep = t[2]-t[1]
        pitch_velocity = np.array([np.diff(pitch, n=1)])
        pitch_velocity = np.transpose(pitch_velocity)/tstep
        pitch_velocity= np.append(pitch_velocity, 0)
        yaw_velocity = np.array([np.diff(yaw, n=1)])
        yaw_velocity = np.transpose(yaw_velocity)/tstep
        yaw_velocity = np.append(yaw_velocity, 0)

        # Compute acceleration derivatives
        # We are computing accelerations becuase we don't trust the measured values are labeled correctly
        d2pitchdt2 = np.array([np.diff(pitch_velocity, n=1)])
        d2pitchdt2 = np.transpose(d2pitchdt2)/tstep
        d2pitchdt2= np.append(d2pitchdt2, 0)
        d2yawdt2 = np.array([np.diff(yaw_velocity, n=1)])
        d2yawdt2 = np.transpose(d2yawdt2)/tstep
        d2yawdt2 = np.append(d2yawdt2, 0)

        inputs = np.array([t, Vp, Vy, pitch, yaw, pitch_velocity, yaw_velocity, d2yawdt2]) # Everything in the order previously defined
        inputs = np.transpose(inputs)

        outputs = np.array([d2pitchdt2])
        outputs = np.transpose(outputs)

        input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, shuffle=False, test_size=0.25)

        model = Sequential([
            Dense(units=500, activation='tanh'),
            Dense(units=200, activation='tanh'),
            Dense(units=100, activation='tanh'),
            Dense(units=5, activation='linear') # Outputs pitch acceleration and a 4 predicted coefficients
        ])

        optm = tf.keras.optimizers.Adam(learning_rate = 0.0001)


        def custom_loss(input_train, output_train, net, itr):
            # Convert inputs and outputs to tensors
            input_train = tf.convert_to_tensor(input_train, dtype=tf.float32)
            output_train = tf.convert_to_tensor(output_train, dtype=tf.float32)

            # Assign an array for the current set of network predictions
            # These are the coefficients that are going to be used in calculating eq. loss
            # and acceleration for collocation loss
            predictions = net(input_train)

        # unpack coefficients
            pred_coeff1 = net(input_train)[:,1]
            pred_coeff2 = net(input_train)[:,2]
            pred_coeff3 = net(input_train)[:,3]
            pred_coeff4 = net(input_train)[:,4]

            # calculate 1st and 2nd numerical derivatives of the coeffiecients
            grad1 = tf.experimental.numpy.diff(pred_coeff1)
            dgrad1 = tf.experimental.numpy.diff(grad1)
            grad2 = tf.experimental.numpy.diff(pred_coeff2)
            dgrad2 = tf.experimental.numpy.diff(grad2)
            grad3 = tf.experimental.numpy.diff(pred_coeff3)
            dgrad3 = tf.experimental.numpy.diff(grad3)
            grad4 = tf.experimental.numpy.diff(pred_coeff4)
            dgrad4 = tf.experimental.numpy.diff(grad4)

            # reshape coeffiecients and their derivatives
            pred_coeff1 = tf.expand_dims(pred_coeff1, axis=1)
            pred_coeff2 = tf.expand_dims(pred_coeff2, axis=1)
            pred_coeff3 = tf.expand_dims(pred_coeff3, axis=1)
            pred_coeff4 = tf.expand_dims(pred_coeff4, axis=1)
            grad1 = tf.expand_dims(grad1, axis=1)
            dgrad1 = tf.expand_dims(dgrad1, axis=1)
            grad2 = tf.expand_dims(grad2, axis=1)
            dgrad2 = tf.expand_dims(dgrad2, axis=1)
            grad3 = tf.expand_dims(grad3, axis=1)
            dgrad3 = tf.expand_dims(dgrad3, axis=1)
            grad4 = tf.expand_dims(grad4, axis=1)
            dgrad4 = tf.expand_dims(dgrad4, axis=1)

            # unpack and reshape the acceleration prediction
            pred_pitch_accel = predictions[:,0]
            pred_pitch_accel = tf.expand_dims(pred_pitch_accel, axis=1)

            # Compute L_c: Collocation point loss
            L_c = tf.reduce_mean(tf.square(output_train - pred_pitch_accel))

            # Getting variables ready for equation loss
            theta = input_train[:,3] # theta = yaw_response
            theta = tf.expand_dims(theta, axis=1)
            dtheta = input_train[:,5] # dtheta = yaw_velocity
            dtheta = tf.expand_dims(dtheta, axis=1)
            Vp = input_train[:,1] # 
            Vp = tf.expand_dims(Vp, axis=1) 
            Vy = input_train[:,2]
            Vy = tf.expand_dims(Vy, axis=1)
            
            # Compute L_eq: Equation Loss
            # This is based on the state space model of the Quanser
            L_eq = tf.reduce_mean(tf.square(pred_coeff1*theta+pred_coeff2*dtheta+pred_coeff3*Vp+pred_coeff4*Vy-output_train)) # Voltage is in Volts

            # Gradient loss: trying to force the coefficients to be constant
            L_gr1 = tf.reduce_mean(tf.square(grad1 - 0)) + tf.reduce_mean(tf.square(dgrad1 - 0))
            L_gr2 = tf.reduce_mean(tf.square(grad2 - 0)) + tf.reduce_mean(tf.square(dgrad2 - 0))
            L_gr3 = tf.reduce_mean(tf.square(grad3 - 0)) + tf.reduce_mean(tf.square(dgrad3 - 0))
            L_gr4 = tf.reduce_mean(tf.square(grad4 - 0)) + tf.reduce_mean(tf.square(dgrad4 - 0))

            L_gr = L_gr1 + L_gr2 + L_gr3 + L_gr4

            # Adjusting loss component weights trying to make them roughly equal
            # Automatically adjust the weights after the first run
            if itr == 0:
                global L_eq_norm, L_gr_norm, L_c_norm
                L_eq_norm = L_eq
                L_gr_norm = L_gr
                L_c_norm = L_c

            L_eq = L_eq/L_eq_norm
            L_gr = L_gr/L_gr_norm
            L_c = L_c/L_c_norm

            L_total = L_eq + L_c + L_gr

            return L_total


        def nrmse(actual_values, predicted_values):
            """
            Calculate Normalized Root Mean Squared Error (NRMSE) for regression. Normalized over the range of the target variable
            In this case it's the range of the pitch acceleration data

            Parameters:
            - actual_values: NumPy array or list of actual values.
            - predicted_values: NumPy array or list of predicted values.

            Returns:
            - nrmse: Normalized Root Mean Squared Error.
            """
            actual_values = np.array(actual_values).flatten()
            predicted_values = np.array(predicted_values).flatten()

            if len(actual_values) != len(predicted_values):
                raise ValueError("Length of actual_values and predicted_values must be the same.")

            # Calculate RMSE
            rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))

            # Calculate the range of the target variable
            target_range = np.max(actual_values) - np.min(actual_values)

            # Calculate NRMSE
            nrmse = (rmse / target_range) * 100 # Expressed as a percentage

            return nrmse


        # Training loop
        train_loss_record = []
        for itr in range(epoch):
            with tf.GradientTape() as tape:
                train_loss = custom_loss(input_train, output_train, model, itr)
                train_loss_record.append(train_loss)
                grad_w = tape.gradient(train_loss, model.trainable_variables)
                optm.apply_gradients(zip(grad_w, model.trainable_variables))

            if itr % 100 == 0:
                print(itr, ': ', train_loss.numpy())


        # Save the model to an HDF5 file
        os.chdir('/home/coder/workspace/VanderWiel/Fennec2023/SystemID/hyper_parameter_study_models')
        model_name = f"learnrate_{i}_epoch_{ii}.h5"
        model.save(model_name)

        # Open the HDF5 file in append mode
        with h5py.File(model_name, 'a') as f:
            # Add metadata attributes
            f.attrs['learning_rate'] = rate
            f.attrs['epochs'] = epoch