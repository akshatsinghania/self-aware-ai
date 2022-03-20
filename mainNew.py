#!/usr/bin/env python3

import sys
import numpy as np
import pickle
from neuron import Brain

def main():
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"will process {num_images} images")
    with open('cifar/data_batch_1', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data']
    labels = data[b'labels']
    nn = Brain(images.shape[1], 256)
    for i,img in enumerate(images[:num_images,::]):
        print(f"#{i}")
        nn.input_image(img)
    nn.generate_random_connections()
    nn.tick()


if '__main__' == __name__:
    main()
