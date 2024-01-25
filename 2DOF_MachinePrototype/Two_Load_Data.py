import pandas as pd
import numpy as np 

def extractData(CSV_List):
    # Create an empty list to store the data
    timesteps = 75
    max_columns = 10
    inputs = np.empty((0, timesteps, max_columns))
    outputs = np.empty((0, 3))

    csv_list = CSV_List


    for dataSet in csv_list:
        data = pd.read_csv(dataSet)

        Time = data.iloc[:, 0].values
        Yaw_Input = data.iloc[:, 1].values  
        Yaw_Response = data.iloc[:, 2].values
        Motor1_Input = data.iloc[:, 3].values
        Motor2_Input = data.iloc[:, 4].values
        Pitch_Input = data.iloc[:, 5].values
        Pitch_Response = data.iloc[:, 6].values
        Accel1 = data.iloc[:,7]
        Accel2 = data.iloc[:,8]
        Motor1_Response = data.iloc[:, 9].values
        Motor2_Response = data.iloc[:, 10].values
        CG = data.iloc[:,11].values
        print('DONE 1')
        # Reset outputData for each dataset
        inputData = []
        outputData = []

        for i in range(timesteps, len(data)):
            timestep_inputs = np.transpose(np.array([Yaw_Input[i-timesteps:i], Yaw_Response[i-timesteps:i], 
            Motor1_Input[i-timesteps:i], Motor2_Input[i-timesteps:i], Pitch_Input[i-timesteps:i], Pitch_Response[i-timesteps:i], 
            Accel1[i-timesteps:i], Accel2[i-timesteps:i], Motor1_Response[i-timesteps:i], Motor2_Response[i-timesteps:i]]))
            inputData.append(timestep_inputs)
            
            # if CG[i] == 0:
            #     outputData.append(np.array([0, 1, 0]))
    
            # elif CG[i] == -1:
            #     outputData.append(np.array([1, 0, 0]))
    
            # elif CG[i] == 1:
            #     outputData.append(np.array([0, 0, 1]))

            # Check conditions based on the dataset name
        if 'Vert' in file_name:
            # Assign [1, 0, 0] to this element of outputData
            outputData.append(np.array([1, 0, 0]))
        elif 'Horiz' in file_name:
            # Assign [0, 0, 1] to this element of outputData
            outputData.append(np.array([0, 0, 1]))
        else:
            # Assign [0, 1, 0] to this element of outputData
            outputData.append(np.array([0, 1, 0]))

        inputs = np.concatenate((inputs, inputData), axis=0)
        outputs = np.concatenate((outputs, outputData), axis=0)
        print('DONE 2')
    return inputs, outputs