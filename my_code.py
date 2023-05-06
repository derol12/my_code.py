import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2

# Define the directory where the face images are stored
data_dir = "D:/img_align_celeba/img_align_celeba"

# Define the size of the images
img_size = (150, 150)

# Create an empty list to store the preprocessed images
data = []

# Loop through all the files in the data directory
for filename in os.listdir(data_dir):
    # Read the image
    img = cv2.imread(os.path.join(data_dir, filename))
    # Resize the image to the specified size
    img = cv2.resize(img, img_size)
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize the pixel values to between 0 and 1
    img = img / 255.0
    # Add the preprocessed image to the data list
    data.append(img)

# Convert the data list to a NumPy array
data = np.array(data)

# Save the data array as a NumPy binary file
np.save("dataset.npy", data)