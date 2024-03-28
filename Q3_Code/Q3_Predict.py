import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from Q3_DataLoader import RNN_load_data

os.chdir('/home/coder/workspace/Data/Becky_Data/')

model_cg = 'CG_Model.h5'
timesteps = 64 # An error will occur if this is not the same as what the model was trained on
data_coeff = 1

# Test Data Sets
#REV 1
dataSet1 = "Norm400Hz_005_AA.csv"
dataSet2 = "Norm400Hz_006_BB.csv" #BAD DATASET - AltHold @ 4m
dataSet3 = "Norm400Hz_007_AA.csv"
dataSet4 = "Norm400Hz_008_AA.csv"
dataSet5 = "Norm400Hz_009_CC.csv" #SHORT
dataSet6 = "Norm400Hz_010_AA.csv"
dataSet7 = "Norm400Hz_014_AA.csv" #BAD DATASET - No 3D display
dataSet8 = "Norm400Hz_015_AA.csv"
dataSet9 = "Norm400Hz_016_BB.csv" #SHORT
dataSet10 = "Norm400Hz_019_AA.csv" #BAD DATASET
dataSet11 = "Norm400Hz_020_CC.csv" #BAD DATASET - Flying at .3m? 
dataSet12 = "Norm400Hz_021_CC.csv"
dataSet13 = "Norm400Hz_022_AA.csv"
dataSet14 = "Norm400Hz_023_AA.csv"
#REV 2
dataSet15 = "Norm400Hz_025_AA.csv" #BAD DATASET - No 3D display
dataSet16 = "Norm400Hz_026_CC.csv"
dataSet17 = "Norm400Hz_027_CC.csv" #DROPOUT
dataSet18 = "Norm400Hz_028_AA.csv" #SHORT
#REV 3
dataSet19 = "Norm400Hz_029_AA.csv" #BAD DATASET
dataSet20 = "Norm400Hz_030_AA.csv"
dataSet21 = "Norm400Hz_031_AA.csv"
dataSet22 = "Norm400Hz_032_AA.csv"
dataSet23 = "Norm400Hz_033_AA.csv"
dataSet24 = "Norm400Hz_034_AA.csv"
dataSet25 = "Norm400Hz_035_AA.csv"
dataSet26 = "Norm400Hz_049_AA.csv"
dataSet27 = "Norm400Hz_051_AA.csv"
dataSet28 = "Norm400Hz_053_AA.csv"
dataSet29 = "Norm400Hz_054_AA.csv"
dataSet30 = "Norm400Hz_055_AA.csv"
dataSet31 = "Norm400Hz_056_AA.csv"
dataSet32 = "Norm400Hz_057_AA.csv"
dataSet33 = "Norm400Hz_058_AA.csv"
dataSet34 = "Norm400Hz_059_AA.csv"
dataSet35 = "Norm400Hz_060_AA.csv"
dataSet36 = "Norm400Hz_061_AA.csv"
dataSet37 = "Norm400Hz_062_BB.csv"
dataSet38 = "Norm400Hz_065_BB.csv"
dataSet39 = "Norm400Hz_066_CC.csv"
dataSet40 = "Norm400Hz_067_BB.csv" #Already trimmed to avoid dropout 
dataSet41 = "Norm400Hz_068_BB.csv"
dataSet42 = "Norm400Hz_069_BB.csv"
dataSet43 = "Norm400Hz_070_CC.csv"
dataSet44 = "Norm400Hz_071_CC.csv"
dataSet45 = "Norm400Hz_072_BB.csv"
dataSet46 = "Norm400Hz_073_BB.csv"
dataSet47 = "Norm400Hz_074_CC.csv"
dataSet48 = "Norm400Hz_075_CC.csv"
dataSet49 = "Norm400Hz_076_CC.csv"

test_data = [dataSet47, dataSet42, dataSet21, #Trained DataSets
             dataSet33, dataSet34, dataSet35, dataSet36, #Unfamiliar DataSets - AA
             dataSet46, # - BB
             dataSet49] # - CC

model = keras.models.load_model(model_cg)

for dataSet in test_data:
    print("Loading Data...", dataSet)
    inputs, outputs = RNN_load_data(dataSet, timesteps, data_coeff)
    predicted = np.array([model.predict(np.array(inputs))])
    predicted = predicted[0]

   # Extracting predicted values
    P = [predicted[:, i] for i in range(3)]

    # Calculating means
    P_means = [np.mean(p) for p in P]

    # Classifying
    categories = ["AA", "BB", "CC"]
    classification = categories[np.argmax(P_means)]
    print("CG Class:", classification, "Actual:", dataSet[14:16])

    # Calculate moving averages
    window_size = 1000
    P_avg = [np.convolve(p, np.ones(window_size) / window_size, mode='valid') for p in P]

    # Plotting
    os.chdir('/home/coder/workspace/Graphs/Q3_Graphs/')
    # plt.style.use("./styles/rose-pine-dawn.mplstyle")
    plt.figure(dpi=300)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))

    for i, (p, avg, mean, label) in enumerate(zip(P, P_avg, P_means, categories)):
        axes[i].plot(range(len(avg)), avg, color='red', label=f"{label} Moving Avg")
        axes[i].scatter(range(len(p)), p, color='blue', marker='o', alpha=0.25, label=f"{label}")
        axes[i].axhline(y=mean, color='orange', linestyle='--', label=f"{label} Mean")
        axes[i].set_title(label)
        axes[i].legend()

    # Adjust the spacing between subplots
    plt.suptitle(f"{dataSet[:-4]} - Predicted Class: {classification}  - data_coeff: {data_coeff} - Timesteps: {timesteps}", fontsize=16)
    plt.tight_layout()

    # Adding labels and title
    plt.savefig(f"{dataSet[:-4]}.png")
    plt.show()

    # Free up memory
    del inputs, outputs, predicted, P, P_means, P_avg, fig, axes