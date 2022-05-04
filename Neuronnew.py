import numpy as np
import scipy.sparse as sparse
import time
from pprint import pprint

#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#a relu activation function
def relu(x):
    return np.maximum(0,x)

#derivative of a relu activation function
def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

class Brain:
    def __init__(self, input_size, depth):
        print(f'Brain({input_size}, {depth})')
        neuron_size = input_size * depth
        self.input_size = input_size
        self.depth = depth
        self.neurons = sparse.lil_array((neuron_size,1),dtype='float32')
        self.weights = sparse.lil_matrix((neuron_size,neuron_size),dtype='float32')
        self.biases = sparse.lil_array((neuron_size,1),dtype='float32')
        self.IDs = set()
        self.activation = relu

    def tick(self):
        print('tick()')
        start = time.perf_counter()
        neurons = self.weights @ self.neurons + self.biases
        self.neurons = sparse.lil_array(self.activation(neurons.todense()))
        end = time.perf_counter()
        print(end-start)

    def input_image(self, image):
        assert self.input_size == np.prod(image.shape)
        print('input_image({})'.format('x'.join([str(x) for x in image.shape])))
        start = time.perf_counter()
        for i,v in enumerate(image.flatten()):
            ID = i*self.depth+v
            if ID not in self.IDs:
                self.add_new_neuron(ID)
            self.neurons[ID] = 1.0
        end = time.perf_counter()
        print("got {} neurons".format(len(self.IDs)))
        print(end-start)

    def generate_random_connections(self):
        print('generate_random_connections()')
        start = time.perf_counter()
        neuron_size = self.input_size * self.depth
        used_neurons = len(self.IDs)
        data = np.random.random(size=(used_neurons**2)).astype(np.float32)
        row = []
        col = []
        for C in self.IDs:
            self.biases[C] = np.random.random()
            for R in self.IDs:
                row.append(R)
                col.append(C)
        self.weights = sparse.csc_matrix((data, (row,col)), shape=(neuron_size,neuron_size))
        self.weights.prune()
        self.neurons = self.neurons.tocsc()
        self.neurons.prune()
        self.biases = self.biases.tocsc()
        self.biases.prune()
        end = time.perf_counter()
        print(end-start)
        pprint(self.neurons)
        pprint(self.weights)

    def add_new_neuron(self, ID):
        # for X in self.IDs:
        #     self.weights[X,ID] = np.random.random()
        #     self.weights[ID,X] = np.random.random()
        self.IDs.add(ID)
