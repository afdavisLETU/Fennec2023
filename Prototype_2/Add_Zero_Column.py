import csv
import os
import numpy as np
import pandas as pd

os.chdir('/home/coder/workspace/Data/Prototype_2_Data/')

# Data Sets
dataSet1 = "Manual_Data1.csv"
dataSet2 = "Manual_Data2.csv"
dataSet3 = "Manual_Data3.csv"
dataSet4 = "Manual_Data4.csv"
dataSet5 = "Manual_Data5.csv"
data = dataSet5

# Define dataframe
df = pd.read_csv(data)

# Read the CSV file into a DataFrame
df.insert(0, 0, 0)
'''
df.insert(column position to be inserted at (0 is far left), 'Name of Column', Number that you want the column to be populated with)
'''

finalname = data.replace('.csv', '_with_Zero.csv')

# Save the DataFrame back to the CSV file
df.to_csv(finalname, index=False)
