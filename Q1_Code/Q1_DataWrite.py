import numpy as np
import pandas as pd
from Q1_DataLoader import get_data, downsample

# Data Sets
dataSet1 = "Low_Wind/006_BB.xlsx"
dataSet2 = "Low_Wind/009_CC.xlsx"
dataSet3 = "Low_Wind/016_BB.xlsx"
dataSet4 = "Low_Wind/020_CC.xlsx"


data = [dataSet1,dataSet2,dataSet3,dataSet4]#,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9]


#Downsample
print("Loading Data...")
for dataSet in data:
    IMU, RCOU = get_data(dataSet)
    IMU_50Hz = downsample(IMU, 8)
    RCOU_50Hz = downsample(RCOU, 8)
    csv_data = np.array(np.hstack((IMU_50Hz, RCOU_50Hz)))
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{dataSet[:-5]}.csv', index=False, header=False)
    print(f"{dataSet[9:-5]}.csv has been created.")