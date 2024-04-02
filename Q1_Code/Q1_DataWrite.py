import os
import numpy as np
import pandas as pd
from Q1_DataProcess import get_data, Micah_Filter, downsample

os.chdir('/home/coder/workspace/Data/Simulator_Data/')

# Data Sets
dataSet1 = "Acro1.xlsx"
dataSet2 = "Acro2.xlsx"
dataSet3 = "Acro3.xlsx"
dataSet4 = "Yaw_Data.xlsx"
dataSet5 = "Flight1.xlsx"
dataSet6 = "Flight2.xlsx"
dataSet7 = "Flight3.xlsx"
dataSet8 = "Normal_Test.xlsx"
dataSet9 = "Flight4.xlsx"

data = [dataSet1,dataSet2,dataSet3]

print("Loading Data...")
for dataSet in data:
    BARO, ATT, GPS, IMU, RCOU = get_data(dataSet)
    IMU = Micah_Filter(IMU, 25, 250, 400)
    GPS = Micah_Filter(GPS, 25, 250, 400)
    BARO = Micah_Filter(BARO, 25, 250, 400)
    ATT = Micah_Filter(ATT, 25, 250, 400)
    RCOU = Micah_Filter(RCOU, 25, 250, 400)
    BARO_20Hz = downsample(BARO, 20)
    GPS_20Hz = downsample(GPS, 20)
    ATT_20Hz = downsample(ATT, 20)
    IMU_20Hz = downsample(IMU, 20)
    RCOU_20Hz = downsample(RCOU, 20)
    csv_data = np.array(np.hstack((BARO_20Hz, ATT_20Hz, GPS_20Hz, IMU_20Hz, RCOU_20Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[:-5]}.csv has been created.")
    

"""
print("Loading Data...")
for i in range(75,500):
    i += 1
    dataSet = f'synthetic_{i:03d}.xlsx'
    BARO, ATT, IMU, RCOU = get_data(dataSet)
    IMU = Micah_Filter(IMU, 25, 250, 400)
    BARO = Micah_Filter(BARO, 25, 250, 400)
    ATT = Micah_Filter(ATT, 25, 250, 400)
    RCOU = Micah_Filter(RCOU, 25, 250, 400)
    BARO_20Hz = downsample(BARO, 10)
    ATT_20Hz = downsample(ATT, 10)
    IMU_20Hz = downsample(IMU, 10)
    RCOU_20Hz = downsample(RCOU, 10)
    csv_data = np.array(np.hstack((IMU_20Hz, BARO_20Hz, ATT_20Hz, RCOU_20Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[:-5]}.csv has been created.")
    """