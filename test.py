import numpy as np


# nclolumn is a list of excite values representing a column of neurons
def update(ncolumn):
    # Create a new list to store the updated values as they are calculated
    updated = []
    # The first neuron has nothing above it, so it must be treated separately
    updated.append((2 * ncolumn[0] + ncolumn[1]) / 3)
    # now loop thrhough the middle ones until just before the end
    for i in range(1, len(ncolumn) - 2):
        updated.append((ncolumn[i - 1] + 2 * ncolumn[i] + ncolumn[i + 1]) / 4)
    # now deal with the one at the end with nothing below it
    updated.append((ncolumn[len(ncolumn) - 2] + 2 * ncolumn[len(ncolumn) - 1]) / 3)
    # replace ncolumn with updated
    return updated


test = np.zeros((10, 10))

print(update(test))
