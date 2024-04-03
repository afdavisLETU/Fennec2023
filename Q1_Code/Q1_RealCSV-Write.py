import os
import numpy as np
import pandas as pd
from Q1_DataProcess import get_real_data, Micah_Filter, downsample

os.chdir('/home/coder/workspace/Data/Real_Data/')

# Data Sets
dataSet1 = "031B.xlsx"
dataSet2 = "032B.xlsx"
dataSet3 = "033B.xlsx"
dataSet4 = "035B.xlsx"
dataSet5 = "051B.xlsx"
dataSet6 = "053B.xlsx"
dataSet7 = "054B.xlsx"
dataSet8 = "055B.xlsx"
dataSet9 = "056B.xlsx"
dataSet10 = "057B.xlsx"
dataSet11 = "059B.xlsx"
dataSet12 = "060B.xlsx"
dataSet13 = "061B.xlsx"
dataSet14 = "063B.xlsx"
dataSet15 = "064B.xlsx"


data = [globals()[f"dataSet{i}"] for i in range(1,16)]

print("Loading Data...")
for dataSet in data:
    BARO, ATT, GPS, IMU, RCOU = get_real_data(dataSet)
    IMU = Micah_Filter(IMU, 200, 100, 400)
    GPS = Micah_Filter(GPS, 200, 100, 400)
    BARO = Micah_Filter(BARO, 200, 100, 400)
    ATT = Micah_Filter(ATT, 200, 100, 400)
    RCOU = Micah_Filter(RCOU, 200, 100, 400)
    BARO = BARO[200:-200]
    IMU = IMU[200:-200]
    GPS = GPS[200:-200]
    ATT = ATT[200:-200]
    RCOU = RCOU[200:-200]
    BARO_20Hz = downsample(BARO, 20)
    GPS_20Hz = downsample(GPS, 20)
    ATT_20Hz = downsample(ATT, 20)
    IMU_20Hz = downsample(IMU, 20)
    RCOU_20Hz = downsample(RCOU, 20)
    csv_data = np.array(np.hstack((BARO_20Hz, ATT_20Hz, GPS_20Hz, IMU_20Hz, RCOU_20Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[:-5]}.csv has been created.")