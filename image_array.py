# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:28:36 2022

@author: abhik
"""

# Let us first import the libraries
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
#import os
#import random
import math
from PIL import Image


class ImageThing:
    

    # A function to change the image into a readable file and save it as an array

    def process_image(self,file_location):
        image = Image.open(file_location)
        image_array = np.asarray(image)
        return image_array
        #For coloured Image it will create 3D array
        #For grayscale Image it will create 2D array




    # Eucleadian Distance: to find distance between center and the given point
    # center: [a0,b0] & point: [a1,b1]
    def find_distance(self,center, point):
        distance = math.sqrt(pow((center[0] - point[0]), 2) + pow((center[1] - point[1]), 2))
        return distance


    # return image in np.array format and 
    def get_data(self,folder_location):
        x = []    # for grayscale image array
        y = []    # distance of center from (1000,1000)
        for i in range(1, 5):
            image_array = self.process_image(folder_location+'//images//' + str(i) + '.jpg')
            x.append(image_array)
            center_point = [len(image_array[1])//2, len(image_array[0])//2]
            distance = self.find_distance(center_point, [1000, 1000])  # center of the image
            y.append(distance)
        return x, y 