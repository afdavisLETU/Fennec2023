import matplotlib.pyplot as plt
from Q3_DataLoader import get_data
from Q3_DataLoader import csv_get_data

def plot_comparison(xlsx_file, csv_file):

    # Load data from xlsx file
    xlsx_IMU, xlsx_RCOU = get_data(xlsx_file)

    # Load data from csv file
    csv_inputs, csv_outputs = csv_get_data(csv_file)

    # Plot IMU data
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xlsx_IMU[:, 0], label='GyrX (xlsx)')
    plt.plot(xlsx_IMU[:, 1], label='GyrY (xlsx)')
    plt.plot(xlsx_IMU[:, 2], label='GyrZ (xlsx)')
    plt.plot(xlsx_IMU[:, 3], label='AccX (xlsx)')
    plt.plot(xlsx_IMU[:, 4], label='AccY (xlsx)')
    plt.plot(xlsx_IMU[:, 5], label='AccZ (xlsx)')
    plt.plot(csv_inputs[:, 0], label='C1 (csv)')
    plt.plot(csv_inputs[:, 1], label='C2 (csv)')
    plt.plot(csv_inputs[:, 2], label='C3 (csv)')
    plt.plot(csv_inputs[:, 3], label='C4 (csv)')
    plt.plot(csv_inputs[:, 4], label='C8 (csv)')
    plt.xlabel('Time')
    plt.ylabel('IMU Data')
    plt.legend()

    # Plot RCOU data
    plt.subplot(1, 2, 2)
    plt.plot(xlsx_RCOU[:, 0], label='C1 (xlsx)')
    plt.plot(xlsx_RCOU[:, 1], label='C2 (xlsx)')
    plt.plot(xlsx_RCOU[:, 2], label='C3 (xlsx)')
    plt.plot(xlsx_RCOU[:, 3], label='C4 (xlsx)')
    plt.plot(xlsx_RCOU[:, 4], label='C8 (xlsx)')

    plt.xlabel('Time')
    plt.ylabel('RCOU Data')
    plt.legend()

    plt.show()

# Call the function with the file paths
xlsx_file = '/home/coder/workspace/Data/Becky_Data/Low_Wind/005_AA.xlsx'
csv_file = '/home/coder/workspace/Data/Becky_Data/Norm400Hz_005_AA.csv'
plot_comparison(xlsx_file, csv_file)

plt.savefig("/home/coder/workspace/Goin/Fennec2023/Q3_Code/TestPlot.png")
plt.show()
