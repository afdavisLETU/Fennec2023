import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Q1_DataProcess import get_real_data, Micah_Filter, downsample

os.chdir('/home/coder/workspace/Data/Real_Data/')

# Data Sets
dataSet1 = "064A.xlsx"

t = 50 * 400

data = [globals()[f"dataSet{i}"] for i in range(1,2)]

print("Loading Data...")
for dataSet in data:
    BARO, ATT, GPS, IMU, RCOU = get_real_data(dataSet)
    BARO_20Hz = downsample(BARO[:t], 20)
    GPS_20Hz = downsample(GPS[:t], 20)
    ATT_20Hz = downsample(ATT[:t], 20)
    IMU_20Hz = downsample(IMU[:t], 20)
    IMU_20Hz = IMU_20Hz[10:-10]
    RCOU_20Hz = downsample(RCOU[:t], 20)
    IMU = Micah_Filter(IMU[:t], 200, 100, 400)
    GPS = Micah_Filter(GPS[:t], 200, 100, 400)
    BARO = Micah_Filter(BARO[:t], 200, 100, 400)
    ATT = Micah_Filter(ATT[:t], 200, 100, 400)
    RCOU = Micah_Filter(RCOU[:t], 200, 100, 400)
    IMU = IMU[200:-200]
    fBARO_20Hz = downsample(BARO, 20)
    fGPS_20Hz = downsample(GPS, 20)
    fATT_20Hz = downsample(ATT, 20)
    fIMU_20Hz = downsample(IMU, 20)
    fRCOU_20Hz = downsample(RCOU, 20)

    # Plotting
    os.chdir('/home/coder/workspace/Graphs/')
    plt.style.use("./styles/rose-pine.mplstyle")
    plt.figure(dpi=300)
    plt.figure(figsize=(10, 6))
    plt.plot(RCOU[:,0], label='Raw')
    plt.plot(fRCOU_20Hz[:,0], label='Filtered')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Data Filtering')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'Filter.png')