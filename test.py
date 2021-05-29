import numpy as np# create dummy data for training


pressure = np.random.rand(3) * 9.9 + 0.1
temperature = np.random.rand(3) * 200 + 273.15
density = pressure * 1e6 / temperature / 8.314 * 1 / 1000

print(pressure.T)
x_train = np.concatenate((pressure, temperature), axis=1)
#x_train = x_train.reshape(-1, 1)
print(x_train)