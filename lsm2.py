import sys
import numpy as np
import scipy.sparse as sparse
import time
import pickle
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
        
    def create_layer(self, layer_size):
        print('create_layer({})'.format(layer_size))
        start = time.perf_counter()
        for i in range(layer_size):
            ID = self.input_size * self.depth + i
            self.add_new_neuron(ID)
        end = time.perf_counter()
        print(end-start)
        
    def create_network(self, network_size):
        print('create_network({})'.format(network_size))
        start = time.perf_counter()
        self.create_layer(network_size[0])
        for i, layer_size in enumerate(network_size[1:]):
            self.create_layer(layer_size)
            #self.create_layer(layer_size)
            #self.create_layer(layer_size)
        end = time.perf_counter()
        print(end-start)

    def visualize(self):
        print('visualize()')
        start = time.perf_counter()
        neuron_size = self.input_size * self.depth
        data = np.zeros(neuron_size**2).astype(np.float32)
        row = []
        col = []
        for C in self.IDs:
            for R in self.IDs:
                row.append(R)
                col.append(C)
                data[R*neuron_size+C] = self.weights[R,C]
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

if __name__ == "__main__":
    #Train the model on CIFAR dataset with 3 connected layers
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"will process {num_images} images")
    with open('cifar/data_batch_1', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data']
    labels = data[b'labels']
    nn = Brain(3, 3) 
    for i in range(num_images):
        #pprint(images[i])
        print(f'label {i} {labels[i]}')
        color_image = images[i][:24].reshape((3,8,2))
        gray_image = images[i][24:].reshape((3,8,4))
        color_image = np.transpose(color_image, (1,2,0))
        gray_image = np.transpose(gray_image, (1,2,0))
        #image = np.hstack((color_image,gray_image))
        image = color_image
        nn.input_image(image)
        pprint(nn.neurons)
        pprint(nn.weights)
        #nn.visualize()
        #break
        #nn.create_network((3,3,3))
        nn.create_network((3,3,3))
        nn.generate_random_connections()
        nn.visualize()
        break
