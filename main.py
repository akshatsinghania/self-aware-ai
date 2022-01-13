from numpy import random
import numpy as np
from timer import call_repeatedly


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


neurons_size = 4
neurons_ids = np.arange(1, 5)
neurons_excites = random.random(size=4)

print("Neurons Size", neurons_size)
print("Neurons Id", neurons_ids)
print("Neurons Excites", neurons_excites)
print()

synapse_size = 4
synapse_hosts = []
synapse_target = []

for i in range(1, neurons_size + 1):
    for x in range(1, neurons_size + 1):
        if i != x:
            synapse_hosts.append(i)
            synapse_target.append(x)

synapse_size = len(synapse_hosts)
synapse_weights = np.random.random(size=synapse_size)
synapse_hebians = np.zeros(synapse_size)


def find_hebbian():
    for indd in range(0, synapse_size):
        host = synapse_hosts[indd]
        target = synapse_target[indd]
        last_average = synapse_hebians[indd]
        new_average = (host + target) / 2
        np.append(synapse_hebians, (last_average + new_average) / 2)


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
