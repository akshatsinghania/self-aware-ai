from numpy import random
import numpy as np
from timer import call_repeatedly


#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#a relu activation function
def relu(x):
    return np.maximum(0,x)

#derivative of a relu activation function
def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

#
neurons_size = 4
neurons_ids = np.arange(0, neurons_size)
neurons_excites = random.random(size=neurons_size)

print("Neurons Size", neurons_size)
print("Neurons Id", neurons_ids)
print("Neurons Excites", neurons_excites)
print()

"""Neurons stacks"""
neurons_stacks_size = neurons_size
neurons_stacks = {}


def update_neurons_stacks(value):
    for key in neurons_stacks:
        neurons_stacks[key] = 0
    neurons_stacks[value] = 1
    sorted_a = []
    for key in neurons_stacks:
        sorted_a.append(key)
    # sorted_a.sort()
    # print("Sorted_a", sorted_a)


print("Stacks size", neurons_stacks_size)
"""Neurons Stacks"""

synapse_size = 4
synapse_hosts = []
synapse_target = []

for i in range(0, neurons_size):
    for x in range(0, neurons_size):
        if i != x:
            synapse_hosts.append(i)
            synapse_target.append(x)

synapse_size = len(synapse_hosts)
synapse_weights = np.random.random(size=synapse_size)
synapse_hebians = np.zeros(synapse_size)

"""Hebbians stacks"""
hebbians_stacks_size = synapse_hebians.size
hebbians_stacks = {}


def update_hebbians_stacks(value):
    for key in hebbians_stacks:
        hebbians_stacks[key] = 0
    hebbians_stacks[value] = 1
    sorted_a = []
    for key in hebbians_stacks:
        sorted_a.append(key)
    # sorted_a.sort()
    # print("Sorted_a", sorted_a)


print("Stacks size", hebbians_stacks_size)
"""Hebbians Stacks"""


# def find_hebbian():
#     for indd in range(0, synapse_size):
#         host = synapse_hosts[indd]
#         target = synapse_target[indd]
#         last_average = synapse_hebians[indd]
#         new_average = (host + target) / 2
#         np.append(synapse_hebians, (last_average + new_average) / 2)
#         update_stacks(indd,new_average)

def find_hebbian():
    neurons_connected_with_values = {}
    for indd in range(0, synapse_size):
        host = neurons_excites[synapse_hosts[indd]]
        target = neurons_excites[synapse_target[indd]]
        last_average = synapse_hebians[indd]
        new_average = (host + target) / 2
        synapse_hebians[indd] = (last_average + new_average) / 2

    for indd in range(0, len(synapse_hosts)):
        neuron_index = synapse_hosts[indd]
        synapse_value = synapse_hebians[indd]
        if neuron_index in neurons_connected_with_values:
            neurons_connected_with_values[neuron_index].append(synapse_value)
        else:
            neurons_connected_with_values[neuron_index] = [synapse_value]
    for avvg in neurons_connected_with_values:
        neurons_excites[avvg] = np.average(neurons_connected_with_values[avvg])
    for values in neurons_excites:
        update_neurons_stacks(values)

    for values in synapse_hebians:
        update_hebbians_stacks(values)


find_hebbian()
call_repeatedly(20, find_hebbian)

print("Synapse size", 4)
print("Synapse Host", synapse_hosts)
print("Synapse Target", synapse_target)
print("Synapse Hebians", synapse_hebians)
print("Synapse Weights", synapse_weights)
print()

perceptron_size = synapse_size
perceptron_outputs = []
perceptron_weights = synapse_weights
perceptron_losses = []

for i in range(0, perceptron_size):
    synapse_weights = np.random.random(size=synapse_size)
    perceptron_weights = synapse_weights

    synapse_host_excite = neurons_excites[synapse_hosts[i] - 1]
    synapse_target_excite = neurons_excites[synapse_target[i] - 1]
    output = (synapse_host_excite * synapse_hebians[i]) + perceptron_weights[i]
    loss = output - synapse_target_excite
    perceptron_outputs.append(output)
    perceptron_losses.append(loss)
    neurons_excites[synapse_target[i] - 1] = output

print("Perceptron Size", perceptron_size)
print("Perceptron Outputs", perceptron_outputs)
print("Perceptron Weights", np.array(perceptron_weights))
print("Perceptron Losses", np.array(perceptron_losses))

class VisionNeuron():
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
        
    def checkActivation(self, neuron):
        for i in range(0, self.input_size):
            for j in range(0, neuron.input_size):
                if self.inputs[i] == neuron.inputs[j]:
                    self.excite = 1
                    neuron.excite = 1
                    return
        self.excite = 0
        neuron.excite = 0
        return
        
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

    #implement a Backprop and Optimizer function for the perceptrons
    def backprop(self, output, target, learning_rate):
        #calculate the loss
        loss = output - target
        #calculate the gradient of the loss
        gradient = sigmoid_derivative(output)
        gradient *= loss
        #update the weights
        for i in range(0, self.input_size):
            self.weights[i] -= learning_rate * gradient * self.inputs[i]
        #update the bias
        self.bias -= learning_rate * gradient
        return loss

    """
        def update(self, inputs, learning_rate):
            #calculate the output
            output = 0
            for j in range(0, self.input_size):
                output += self.inputs[j] * self.weights[j]
            output += self.bias
            self.output = sigmoid(output)
            #calculate the loss
            loss = output - target
            #calculate the gradient of the loss
            gradient = sigmoid_derivative(output)
            gradient *= loss
            #update the weights
            for i in range(0, self.input_size):
                self.weights[i] -= learning_rate * gradient * self.inputs[i]
            #update the bias
            self.bias -= learning_rate * gradient
            return loss
    """
    def optimizer(self, inputs, target, learning_rate, epochs):
        for i in range(0, epochs):
            loss = self.backprop(inputs, target, learning_rate)
            # print("Loss", loss)
        return loss
    
    def sigmoid_derivative(self, output):
        return output * (1 - output)
        
#a function that checks every value passed to it by the vision function and if new form a new neuron and load it to that neuron
def vision_function(inputs, input_size):
    vision_neurons = []
    #check if the inputs are in the vision neurons array
    for i in range(0, len(vision_neurons)):
        if vision_neurons[i].inputs == inputs:
            return vision_neurons[i]
    #if the inputs are not in the vision neurons array then create a new neuron
    vision_neurons.append(VisionNeuron(inputs, input_size))
    return vision_neurons[-1]
    
def Feeder(inputs, input_size):
    neurons = vision_function(inputs, input_size)
    for i in range(0, input_size):
        for j in range(0, input_size):
            neurons[i].checkActivation(neurons[j])
    return neurons
    
def Feeder_with_weights(inputs, input_size):
    neurons = Feeder(inputs, input_size)
    for i in range(0, input_size):
        for j in range(0, input_size):
            if neurons[i].excite == 1 and neurons[j].excite == 1:
                neurons[i].recalculateWeights()
                neurons[j].recalculateWeights()
                neurons[i].recalcuateBias()
                neurons[j].recalcuateBias()
                neurons[i].optimizer(inputs, inputs[j], 0.1, 100)
                neurons[j].optimizer(inputs, inputs[i], 0.1, 100)
                # print("Neuron", i, "Weights", neurons[i].weights)
                # print("Neuron", j, "Weights", neurons[j].weights)
    return neurons
    
def NeuralNet(inputs, input_size):
    neurons = Feeder_with_weights(inputs, input_size)
    for i in range(0, input_size):
        neurons[i].excite = 0
        neurons[i].calculateOutput(inputs=inputs)
        # print("Neuron", i, "Excite", neurons[i].excite)
        # print("Neuron", i, "Output", neurons[i].output)
    return neurons
    
def FeedForward(x, input_size):
    return NeuralNet(x, input_size)
    
def BackPropagation(x, input_size):
    return Feeder_with_weights(x, input_size)

def main():
    #using cifar dataset. the data set has 32x32 pixles in 3 channels
    inputs = np.random.randint(0,255, size=(1000,32,32,3))
    for i in range(0, inputs.shape[0]):
        for layer in range(0, inputs.shape[3]):
            inputs[i, :, :, layer] = FeedForward(inputs[i, :, :, layer], inputs.shape[2])
    print("Inputs", inputs)
    print("Inputs Shape", inputs.shape)
    
    targets = np.random.randint(0,255, size=(1000,32,32,3))
    for i in range(0, targets.shape[0]):
        for layer in range(0, targets.shape[3]):
            targets[i, :, :, layer] = BackPropagation(targets[i, :, :, layer], inputs.shape[2])
    print("Targets", targets)
    print("Targets Shape", targets.shape)

if __name__ == '__main__':
    main()
