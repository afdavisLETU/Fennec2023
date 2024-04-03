import os
import numpy as np
import pandas as pd
from Q1_DataProcess import get_sim_data, Micah_Filter, downsample

os.chdir('/home/coder/workspace/Data/Simulator_Data/')

# Data Sets
dataSet1 = "Acro1.xlsx"
dataSet2 = "Acro2.xlsx"
dataSet3 = "Acro3.xlsx"
dataSet4 = "Acro_Test.xlsx"
dataSet5 = "X_Data.xlsx"
dataSet6 = "Y_Data.xlsx"
dataSet7 = "Z_Data.xlsx"
dataSet8 = "Yaw_Data.xlsx"
dataSet9 = "Flight1.xlsx"
dataSet10 = "Flight2.xlsx"
dataSet11 = "Flight3.xlsx"
dataSet12 = "Flight4.xlsx"
dataSet13 = "Flight5.xlsx"
dataSet14 = "Normal_Test.xlsx"
dataSet15 = "Drift1.xlsx"
dataSet16 = "Drift2.xlsx"
dataSet17 = "Acro4.xlsx"
dataSet18 = "Y_Data2.xlsx"
dataSet19 = "Yaw_Data2.xlsx"
dataSet20 = "Acro5.xlsx"
dataSet21 = "Flight6.xlsx"
dataSet22 = "Flight7.xlsx"

data = [globals()[f"dataSet{i}"] for i in range(22, 23)]

print("Loading Data...")
for dataSet in data:
    BARO, ATT, GPS, IMU, RCOU = get_sim_data(dataSet)
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