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
dataSet50 = "Low_Wind/Exp_030_AA.xlsx"
dataSet51 = "Low_Wind/Exp_031_AA.xlsx"
dataSet52 = "Low_Wind/Exp_032_AA.xlsx"
dataSet53 = "Low_Wind/Exp_033_AA.xlsx"
dataSet54 = "Low_Wind/Exp_035_AA.xlsx"
dataSet55 = "Low_Wind/Exp_051_AA.xlsx"
dataSet56 = "Low_Wind/Exp_054_AA.xlsx"
dataSet57 = "Low_Wind/Exp_056_AA.xlsx"
dataSet58 = "Low_Wind/Exp_062_BB.xlsx"
dataSet59 = "Low_Wind/Exp_065_BB.xlsx"
dataSet60 = "Low_Wind/Exp_066_CC.xlsx"
dataSet61 = "Low_Wind/Exp_067_BB.xlsx"
dataSet62 = "Low_Wind/Exp_068_BB.xlsx"
dataSet63 = "Low_Wind/Exp_069_BB.xlsx"
dataSet64 = "Low_Wind/Exp_070_CC.xlsx"
dataSet65 = "Low_Wind/Exp_071_CC.xlsx"
dataSet66 = "Low_Wind/Exp_072_BB.xlsx"
dataSet67 = "Low_Wind/Exp_073_BB.xlsx"
dataSet68 = "Low_Wind/Exp_074_CC.xlsx"
dataSet69 = "Low_Wind/Exp_075_CC.xlsx"
dataSet70 = "Low_Wind/Exp_076_CC.xlsx"

data = [dataSet50, dataSet51, dataSet52, dataSet53, dataSet54, dataSet55, dataSet56, dataSet57, dataSet58, dataSet59, dataSet60, dataSet61, dataSet62, dataSet63, dataSet64, dataSet65, dataSet66, dataSet67, dataSet68, dataSet69, dataSet70]

print("Loading Data...")
for dataSet in data:
    IMU, RCOU, ATT = get_data(dataSet)
    csv_data = np.array(np.hstack((IMU, RCOU, ATT)))
    df = pd.DataFrame(csv_data)
    file_name = os.path.basename(dataSet)  # Get the file name without the directory path
    df.to_csv(f'Norm400Hz_{file_name[:-5]}.csv', index=False, header=False)
    print(f"Norm400Hz_{dataSet[9:-5]}.csv has been created.")
