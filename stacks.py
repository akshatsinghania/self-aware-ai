import numpy as np

stacks_size = 10

stacks = np.zeros(stacks_size)


def update_stacks(value, index):
    stacks[index]=value
    current_value = value
    for i in range(index-1, 0, -1):
        current_value -= 1
        stacks[i] = current_value

    current_value = value
    for i in range(index+1, stacks_size):
        current_value -= 1
        stacks[i] = current_value



