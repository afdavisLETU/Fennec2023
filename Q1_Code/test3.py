import numpy as np
import matplotlib.pyplot as plt
from Q1_DataLoader import get_data

IMU, RCOU = get_data("Low_Wind/010_AA.xlsx")

x = range(len(IMU[:50000,3]))
x = np.array(x) / 400
plt.plot(x, IMU[:50000,3])
plt.plot(x, stuff[:50000,3],'r--')
plt.title("Micah Filter Test")
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig('Test4.png')
plt.show()



