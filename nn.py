from numpy import random
import math
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, inputs, input_size, bias=None):
        self.input_size = input_size
        self.inputs = inputs
        if bias:
            self.bias = bias
        else:
            self.bias = random.rand()
        self.weights = random.random(size=input_size)
        self.calculateOutput(inputs=self.inputs)
        # print("Bias ", self.bias)
        # print("Weights ", self.weights)
        # print("Output ", self.output)

    def recalculateWeights(self):
        self.weights = random.random(size=self.input_size)

    def recalcuateBias(self):
        self.bias = random.rand()

    def calculateOutput(self, inputs):
        self.inputs = inputs
        output = 0
        for j in range(0, self.input_size):
            output += self.inputs[j] * self.weights[j]
        output += self.bias
        self.output = sigmoid(output)


"""
[0,1]-> 1
[1,1]-> 0
"""

inputs_layer = [[1, 0], [0, 0], [1, 1], [0, 1]]
out_desired = [1, 0, 0, 1]
neurons = [Neuron(inputs=inputs_layer[0], input_size=len(inputs_layer[0]))]

# loss = 100
best = 0

k = 0

while True:
    break_w = True
    correct = 0
    neurons[0].recalcuateBias()
    neurons[0].recalculateWeights()
    for index, inputs in inputs_layer:
        neurons[0].calculateOutput(inputs=inputs_layer[index])
        output = 0
        if neurons[0].output >= 0.5:
            output = 1
        else:
            output = 0
        if output == out_desired[index]:
            correct += 1
    print(correct)

print("Best bias", best)
