import queue

import numpy as np
import functools
import matplotlib.pyplot as plt

max_x = 10
max_y = 10
image_neurons = np.zeros((max_x, max_y))


# Function to check if a cell
# is be visited or not
def isValid(vis, row, col):
    # If cell lies out of bounds
    if row < 0 or col < 0 or row >= max_x or col >= max_y:
        return False

    # If cell is already visited
    if vis[row][col]:
        return False

    # Otherwise
    return True


def fire_neuron(grid, row, col):
    vis = np.zeros((max_x, max_y))
    # Stores indices of the matrix cells
    q = []
    dRow = [0, 1, 1, -1, 0, -1, 1]
    dCol = [1, 0, 1, 0, -1, -1, 1]
    # Mark the starting cell as visited
    # and push it into the queue
    q.append((row, col, max(row, col, max_x - row, max_y - col)))
    vis[row][col] = True

    # Iterate while the queue
    # is not empty
    while len(q) > 0:
        cell = q.pop()
        x = cell[0]
        y = cell[1]
        value = cell[2]
        image_neurons[x][y] = value
        # q.pop()

        # Go to the adjacent cells
        for i in range(len(dCol)):
            adjx = x + dRow[i]
            adjy = y + dCol[i]
            if isValid(vis, adjx, adjy):
                q.append((adjx, adjy, value - 1))
                print((adjx, adjy, value - 1), end='\t')
                vis[adjx][adjy] = True
        print()


# print("Image neurons", len(image_neurons))
# fire_neuron(image_neurons, 0, 0)
# print(image_neurons)
# plt.imshow(image_neurons, cmap='gray', interpolation='none')
# plt.show()

value = 4
dRow = [0, 1, 1, -1, 0]
dCol = [1, 0, 1, 0, -1]
x = 3
y = 3
vis = np.zeros((max_x, max_y))
for i in range(len(dCol)):
    adjx = x + dRow[i]
    adjy = y + dCol[i]
    image_neurons[adjx][adjy] = value - 1
    if isValid(vis, adjx, adjy):
        image_neurons[adjx][adjy] = value - 1
        print((adjx, adjy, value - 1), end='\t')

plt.imshow(image_neurons, interpolation='none')
plt.show()
