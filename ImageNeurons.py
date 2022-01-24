import queue

import numpy as np
import matplotlib.pyplot as plt

max_x = 30
max_y = 30
image_neurons = np.zeros((max_x, max_y))

directions = [[0, 1], [1, 0],
              [0, -1], [-1, 0],
              [1, 1], [-1, -1],
              [-1, 1], [1, -1],
              ]

old_images=[]

def fire_neuron(col, row):
    old_images.append(image_neurons)
    visited = np.zeros((max_x, max_y))
    q = queue.Queue()
    value = max(col, row, max_x - col, max_y - row)
    q.put([col, row, value])
    while not q.empty():
        current = q.get()
        x = current[0]
        y = current[1]
        value = current[2]
        image_neurons[x][y] = value
        for direction in directions:
            new_x = x + direction[0]
            new_y = y + direction[1]
            if 0 <= new_x < max_x and 0 <= new_y < max_y and not visited[new_x][new_y]:
                q.put([new_x, new_y, value - 1])
                visited[new_x][new_y] = 1


fire_neuron(20, 15)
plt.imshow(image_neurons,cmap='gray', interpolation='none')
plt.show()
print(old_images)
