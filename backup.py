'''this is a neuron
holds links to other neurons
modifies link strengths according to conditions
'''
import random
import time

from tqdm import tqdm
import numpy as np
from timer import call_repeatedly


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Brain:
    def __init__(self, size, input_size, output_size, num_synapses=None, density=None):
        assert num_synapses or density
        assert not (num_synapses and density)

        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.density = density
        self.num_synapses = num_synapses
        if density:
            self.num_synapses = int(density * size)
        self.num_synapses = num_synapses if not density else int(density * size)

        self.input_neurons = np.random.randint(0, size, input_size, dtype=int)
        self.output_neurons = np.random.randint(0, size, output_size, dtype=int)

        self.neurons = np.arange(size, dtype=float)

        # print(f"num_synapses: {self.num_synapses}")
        # print(f"synaptic size: {size * self.num_synapses})")
        self.synapses = np.random.randint(0, size, (size, self.num_synapses), dtype=int)
        self.strengths = np.ones((size, self.num_synapses), dtype=float)
        self.hebbians = np.zeros((size, self.num_synapses), dtype=float)

    def step(self):
        self.step_neurons()
        self.step_synapses()

    def step_neurons(self):
        self.neurons = sigmoid(np.sum(self.strengths, axis=1) / self.num_synapses)

    def step_synapses(self):
        self.strengths = self.hebbians * (self.neurons + self.neurons[self.synapses].T).T / 2

    def store(self, data):
        self.neurons[self.input_neurons] = data

    def load(self):
        return self.neurons[self.output_neurons]

    def update_hebbians(self, value):
        self.hebbians = value

    def __repr__(self) -> str:
        return f"Brain: size: {self.size}, density: {self.density}"


class Hebcal:
    def __init__(self, host, target):
        self.host = host
        self.target = target
        self.value = (target.hebbians + host.hebbians) / 2

    def recalculate(self, host, target):
        self.host = host
        self.target = target
        new_value = (host.hebbians + target.hebbians) / 2
        current_value = (new_value + self.value) / 2
        self.value = current_value


brains = [Brain(size=8, num_synapses=4, input_size=4, output_size=4),
          Brain(size=8, num_synapses=4, input_size=4, output_size=4)]
hebcals = []
hebcals_index = []


def check_for_hebcals_update():
    for i in range(0, len(hebcals_index)):
        hebcals[i].recalculate(brains[hebcals_index[i][0]], brains[hebcals_index[i][1]])
        print("Recalculated Value")
        print(hebcals[i].value)


def get_brains_and_hebcals(size):
    for i in range(1, size + 1):
        brains.append(Brain(size=8, num_synapses=4, input_size=4, output_size=4))
        if i > 1 and i < size:
            hebcals_index.append([i - 1, i])
        brains[0].update_hebbians(np.random.random(brains[0].input_size))

    for i in hebcals_index:
        hebcals.append(Hebcal(host=brains[i[0]], target=brains[i[1]]))

    print(len(hebcals), " hebcals created")
    for i in hebcals:
        print(i.value)

    call_repeatedly(20, check_for_hebcals_update)


if __name__ == "__main__":
    brain = Brain(size=8, num_synapses=4, input_size=4, output_size=4)

    data = np.random.random(brain.input_size)
    print(brain.neurons)
    # print(data)
    # brain.store(data)
    # for _ in range(10):
    #     brain.step()
    #     print(brain.load())

    #get_brains_and_hebcals(10)
