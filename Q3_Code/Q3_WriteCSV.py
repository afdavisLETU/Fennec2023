import os
import numpy as np
import pandas as pd
from Q3_DataLoader import get_data

os.chdir('/home/coder/workspace/Data/Becky_Data/')

# Data Sets
dataSet1 = "Low_Wind/005_AA.xlsx"
dataSet2 = "Low_Wind/006_BB.xlsx"
dataSet3 = "Low_Wind/007_AA.xlsx"
dataSet4 = "Low_Wind/008_AA.xlsx"
dataSet5 = "Low_Wind/009_CC.xlsx"
dataSet6 = "Low_Wind/010_AA.xlsx"
dataSet7 = "Low_Wind/014_AA.xlsx"
dataSet8 = "Low_Wind/015_AA.xlsx"
dataSet9 = "Low_Wind/016_BB.xlsx"
dataSet10 = "Low_Wind/019_AA.xlsx"
dataSet11 = "Low_Wind/020_CC.xlsx"
dataSet12 = "Low_Wind/021_CC.xlsx"
dataSet13 = "Low_Wind/022_AA.xlsx"
dataSet14 = "Low_Wind/023_AA.xlsx"
dataSet15 = "Low_Wind/025_AA.xlsx"
dataSet16 = "Low_Wind/026_CC.xlsx"
dataSet17 = "Low_Wind/027_CC.xlsx"
dataSet18 = "Low_Wind/028_AA.xlsx"

data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8,dataSet9,dataSet10,dataSet11,dataSet12,dataSet13,dataSet14,dataSet15,dataSet16,dataSet17,dataSet18]

print("Loading Data...")
for dataSet in data:
    IMU, RCOU = get_data(dataSet)
    csv_data = np.array(np.hstack((IMU, RCOU)))
    df = pd.DataFrame(csv_data)
    file_name = os.path.basename(dataSet)  # Get the file name without the directory path
    df.to_csv(f'Norm400Hz_{file_name[:-5]}.csv', index=False, header=False)
    print(f"Norm400Hz_{dataSet[9:-5]}.csv has been created.")
