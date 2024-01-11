import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Function to remove noise using a Fourier transform
def high_pass(data, noise_frequency, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_or(freq > noise_frequency, freq < -noise_frequency)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real  # Extract real values

def low_pass(data, noise_frequency, sampling_rate):
    n = len(data)
    fft_data = fft(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    noise_mask = np.logical_and(freq < noise_frequency, freq > -noise_frequency)
    clean_data = ifft(fft_data * noise_mask)
    return clean_data.real  # Extract real values

# Load accelerometer data from a CSV file
def load_accelerometer_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    ms = data[:, 1]
    y = data[:, 2]
    p = data[:, 3]
    r = data[:, 4]
    return ms, y, p, r

# Specify the CSV file path and noise frequency
file_path = "AutoP2_data6.csv"
output_file_path = "cleaned_acceleration_data.csv"
low_noise = 0.015
high_noise = 2

# Load accelerometer data
ms, y, p, r = load_accelerometer_data(file_path)

# Sampling rate
sampling_rate = 25
end_time = len(ms)/25
time = np.arange(0, end_time, 1/sampling_rate)

# Remove noise using the specified noise frequency
clean_ms = high_pass(ms, low_noise, sampling_rate)
clean_y = high_pass(y, low_noise, sampling_rate)
clean_p = high_pass(p, low_noise, sampling_rate)
clean_r = high_pass(r, low_noise, sampling_rate)

clean_ms = low_pass(clean_ms, high_noise, sampling_rate)
clean_y = low_pass(clean_y, high_noise, sampling_rate)
clean_p = low_pass(clean_p, high_noise, sampling_rate)
clean_r = low_pass(clean_r, high_noise, sampling_rate)

# Save cleaned acceleration data (real values) to a new CSV file
cleaned_data = np.column_stack((time, clean_r))
np.savetxt(output_file_path, cleaned_data, delimiter=',', header='Time, Cleaned Acceleration (Real Values)', comments='')

# Plot the original and cleaned data
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, r)
plt.title("Original Acceleration Data")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")

plt.subplot(3, 1, 2)
plt.plot(time, clean_r)
plt.title("Cleaned Acceleration Data")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")

# Plot the magnitude spectrum for original data
plt.subplot(3, 1, 3)
original_spectrum = np.abs(fft(r))
freq = np.fft.fftfreq(len(r), d=1/sampling_rate)
plt.plot(freq, original_spectrum)
plt.title("Magnitude Spectrum (Original Data)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim([-1, 1])

plt.tight_layout()
plt.show()
