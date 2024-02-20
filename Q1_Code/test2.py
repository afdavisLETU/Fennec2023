import os
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def low_pass(data, noise_frequency, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_and(freq < noise_frequency, freq > -noise_frequency)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real

# Define the dimensions of the array
data_length = 750
window = 15

# Generate a 2D array of random numbers
zeros = 250
data = np.zeros(zeros)
for i in range(data_length):
    data = np.append(data, data[i+zeros-1]*0.9+np.random.uniform(-1, 1)*0.1)

old_filter = low_pass(data, 10, 400)

clean_data = []
for i in range(len(data)-window+1):
    clean_data.append(low_pass(data[i:window+i], 10, 400))
clean_data = np.array(clean_data)

# Compute the diagonals
new_filter = []
for i in range(window-1):
    new_filter.append(clean_data[0,i])
for r in range(len(clean_data) - window + 1):
    diagonal_values = clean_data[r:r + window, window - np.arange(window) - 1]
    new_filter.append(np.mean(diagonal_values))
for i in range(window-1):
    new_filter.append(clean_data[len(clean_data)-1,i+1])

x = range(len(data))
plt.plot(x, data)
plt.plot(x, old_filter)
#plt.plot(x, new_filter,'r--')
plt.title("Heikkila Filter Test")
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
os.chdir('/home/coder/workspace/Data/Becky_Data/')
plt.savefig('Test.png')
plt.show()
