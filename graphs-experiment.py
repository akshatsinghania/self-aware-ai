import numpy as np
import matplotlib.pyplot as plt

neurons = np.random.random(size=(10,10))

plt.imshow(neurons,interpolation='none')
plt.show()