import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from Q3_DataLoader import RNN_load_data

model_cg = 'CG_Model.h5'
timesteps = 750
data_coeff = 0.15

# Test Data Sets
dataSet1 = "AA_Data1.xlsx"
dataSet2 = "BB_Data1.xlsx"
dataSet3 = "CC_Data1.xlsx"
dataSet4 = "AA_Data2.xlsx"
dataSet5 = "AA_Data3.xlsx"

test_data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5]

for dataSet in test_data:
    print("Loading Data...")
    inputs, outputs = RNN_load_data(dataSet, timesteps, data_coeff)
    model = keras.models.load_model(model_cg)
    predicted = np.array([model.predict(np.array(inputs))])
    predicted = predicted[0]

   # Extracting predicted values
    P = [predicted[:, i] for i in range(3)]

    # Calculating means
    P_means = [np.mean(p) for p in P]

    # Classifying
    categories = ["AA", "BB", "CC"]
    classification = categories[np.argmax(P_means)]
    print("CG Class:", classification)

    # Calculate moving averages
    window_size = 1000
    P_avg = [np.convolve(p, np.ones(window_size) / window_size, mode='valid') for p in P]

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))

    for i, (p, avg, mean, label) in enumerate(zip(P, P_avg, P_means, categories)):
        axes[i].plot(range(len(avg)), avg, color='red', label=f"{label} Moving Avg")
        axes[i].scatter(range(len(p)), p, color='blue', marker='o', alpha=0.25, label=f"{label}")
        axes[i].axhline(y=mean, color='orange', linestyle='--', label=f"{label} Mean")
        axes[i].set_title(label)
        axes[i].legend()

    # Adjust the spacing between subplots
    plt.suptitle(f"{dataSet[:-5]} Predicted Class: {classification}", fontsize=16)
    plt.tight_layout()

    # Adding labels and title
    plt.savefig(f"{dataSet[:-5]}.png")
    plt.show()