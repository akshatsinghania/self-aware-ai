from numpy import random
import numpy as np
from timer import call_repeatedly
import ImageThing as ia


#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#
neurons_size = 4
neurons_ids = np.arange(0, neurons_size)
neurons_excites = random.randint(0,size=neurons_size)

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
