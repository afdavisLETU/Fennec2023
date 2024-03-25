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
dataSet19 = "Low_Wind/029_AA.xlsx"
dataSet20 = "Low_Wind/030_AA.xlsx"
dataSet21 = "Low_Wind/031_AA.xlsx"
dataSet22 = "Low_Wind/032_AA.xlsx"
dataSet23 = "Low_Wind/033_AA.xlsx"
dataSet24 = "Low_Wind/034_AA.xlsx"
dataSet25 = "Low_Wind/035_AA.xlsx"
dataSet26 = "Low_Wind/049_AA.xlsx"
dataSet27 = "Low_Wind/051_AA.xlsx"
dataSet28 = "Low_Wind/053_AA.xlsx"
dataSet29 = "Low_Wind/054_AA.xlsx"
dataSet30 = "Low_Wind/055_AA.xlsx"
dataSet31 = "Low_Wind/056_AA.xlsx"
dataSet32 = "Low_Wind/057_AA.xlsx"
dataSet33 = "Low_Wind/058_AA.xlsx"
dataSet34 = "Low_Wind/059_AA.xlsx"
dataSet35 = "Low_Wind/060_AA.xlsx"
dataSet36 = "Low_Wind/061_AA.xlsx"
dataSet37 = "Low_Wind/062_BB.xlsx"
dataSet38 = "Low_Wind/065_BB.xlsx"
dataSet39 = "Low_Wind/066_CC.xlsx"
dataSet40 = "Low_Wind/067_BB.xlsx"
dataSet41 = "Low_Wind/068_BB.xlsx"
dataSet42 = "Low_Wind/069_BB.xlsx"
dataSet43 = "Low_Wind/070_CC.xlsx"
dataSet44 = "Low_Wind/071_CC.xlsx"
dataSet45 = "Low_Wind/072_BB.xlsx"
dataSet46 = "Low_Wind/073_BB.xlsx"
dataSet47 = "Low_Wind/074_CC.xlsx"
dataSet48 = "Low_Wind/075_CC.xlsx"
dataSet49 = "Low_Wind/076_CC.xlsx"


data = [dataSet21, dataSet22, dataSet23, dataSet24, dataSet25, dataSet26, dataSet27, dataSet28, dataSet29, dataSet30, dataSet31, dataSet32, dataSet33, dataSet34, dataSet35, dataSet36]

print("Loading Data...")
for dataSet in data:
    IMU, RCOU = get_data(dataSet)
    csv_data = np.array(np.hstack((IMU, RCOU)))
    df = pd.DataFrame(csv_data)
    file_name = os.path.basename(dataSet)  # Get the file name without the directory path
    df.to_csv(f'Norm400Hz_{file_name[:-5]}.csv', index=False, header=False)
    print(f"Norm400Hz_{dataSet[9:-5]}.csv has been created.")
