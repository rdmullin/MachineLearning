import numpy as np
import matplotlib.pyplot as plt
import random

#Generate Data Samples
X = np.zeros(10)
for i in range(10):
    X[i] = random.randint(1,100)
    X[i] = (X[i] * 2) + 50

    

