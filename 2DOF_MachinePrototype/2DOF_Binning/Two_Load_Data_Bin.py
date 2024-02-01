import pandas as pd
import numpy as np 

def extractData(CSV_List):
    # Create an empty list to store the data
    timesteps = 75
    max_columns = 2
    inputs = np.empty((0, timesteps, max_columns))
    outputs = np.empty((0, 2))

    csv_list = CSV_List


    for dataSet in csv_list:
        data = pd.read_csv(dataSet)

        PitchRad = data.iloc[:, 0].values
        YawRad = data.iloc[:, 1].values  
        Pitch_Dot = data.iloc[:, 2].values
        Yaw_Dot = data.iloc[:, 3].values
        DesiredPitch = data.iloc[:, 4].values
        DesiredYaw = data.iloc[:, 5].values
        Motor0CurrentRAW = data.iloc[:, 6].values
        Motor1CurrentRAW = data.iloc[:,7].values
        TimeRAW = data.iloc[:,8].values

        Motor0Current = Motor0CurrentRAW / 5000
        Motor1Current = Motor1CurrentRAW / 5000
        Time = TimeRAW / 10000 - 1

        # Reset outputData for each dataset
        inputData = []
        outputData = np.zeros((len(data) - timesteps, 2))  # Initialize as a numpy array

        for i in range(timesteps, len(data)):
            timestep_inputs = np.transpose(np.array([PitchRad[i-timesteps:i], YawRad[i-timesteps:i]]))#, 
            # Pitch_Dot[i-timesteps:i], Yaw_Dot[i-timesteps:i], 
            # DesiredPitch[i-timesteps:i], DesiredYaw[i-timesteps:i], 
            # Motor0Current[i-timesteps:i], Motor1Current[i-timesteps:i], 
            #Time[i-timesteps:i]]))
            inputData.append(timestep_inputs)

            # Check conditions based on the dataset name
        if 'Z' in dataSet:
            # Assign [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
            outputData[:, 0] = 1
        elif 'X' in dataSet:
            # Assign [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
            outputData[:, 1] = 1
        elif '3' in dataSet:
            # Assign [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] to this element of outputData
            outputData[:, 2] = 1 
        elif '4' in dataSet:
            # Assign [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] to this element of outputData
            outputData[:, 3] = 1  
        elif '5' in dataSet:
            # Assign [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] to this element of outputData
            outputData[:, 4] = 1
        elif '6' in dataSet:
            # Assign [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] to this element of outputData
            outputData[:, 5] = 1
        elif '7' in dataSet:
            # Assign [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] to this element of outputData
            outputData[:, 6] = 1
        elif '8' in dataSet:
            # Assign [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] to this element of outputData
            outputData[:, 7] = 1
        elif '9' in dataSet:
            # Assign [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] to this element of outputData
            outputData[:, 8] = 1
        else:
            # Assign [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] to this element of outputData
            outputData[:, 1] = 1

        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
        print(f'Outputs assigned for {dataSet}')
    # print(outputs.shape)
    # print(inputs.shape)
    return inputs, outputs