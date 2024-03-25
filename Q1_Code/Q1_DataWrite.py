import os
import numpy as np
import pandas as pd
from Q1_DataProcess import get_data, Micah_Filter, downsample

os.chdir('/home/coder/workspace/Data/Synthetic_Data/')

print("Loading Data...")
for i in range(103,500):
    i += 1
    dataSet = f'synthetic_{i:03d}.xlsx'
    BARO, ATT, IMU, RCOU = get_data(dataSet)
    #IMU = Micah_Filter(IMU, 25, 250, 50)
    BARO_20Hz = downsample(BARO, 20)
    ATT_20Hz = downsample(ATT, 20)
    IMU_20Hz = downsample(IMU, 20)
    RCOU_20Hz = downsample(RCOU, 20)
    csv_data = np.array(np.hstack((BARO_20Hz, IMU_20Hz, ATT_20Hz, RCOU_20Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[:-5]}.csv has been created.")

"""
# Data Sets
dataSet1 = "Low_Wind/006_BB.xlsx"
dataSet2 = "Low_Wind/009_CC.xlsx"
dataSet3 = "Low_Wind/016_BB.xlsx"
dataSet4 = "Low_Wind/020_CC.xlsx"

data = [dataSet1,dataSet2,dataSet3,dataSet4]

print("Loading Data...")
for dataSet in data:
    IMU, RCOU = get_data(dataSet)
    IMU_50Hz = downsample(IMU, 8)
    RCOU_50Hz = downsample(RCOU, 8)
    csv_data = np.array(np.hstack((IMU_50Hz, RCOU_50Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[9:-5]}.csv has been created.")
    """