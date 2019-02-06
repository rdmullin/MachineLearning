import numpy as np
import matplotlib.pyplot as plt

#Generate Data Samples

x = np.random.randint(0,100,10)
y = (x * 2) + 50

#Plot the landscape of the loss function

bb = np.arange(0,100,1)
ww = np.arange(-5,5,0.1)
Z = np.zeros((len(bb),len(ww)))

for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[i]
        Z[j][i] = 0
        
        for n in range(len(x)):
            Z[j][i] = Z[j][i] + (y[n] - b - w*x[n]**2)
            Z[j][i] = Z[j][i]/len(x)
plt.contour(bb,ww,Z,50, alpha = 0.5, cmap = plt.get_cmap('jet'))
plt.plot([50],[2],'x', ms = 12, markeredgewidth = 3, color= 'orange')
plt.xlim(0,100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()