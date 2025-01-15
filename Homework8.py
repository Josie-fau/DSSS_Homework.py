from tensorflow import keras

keras.datasets.mnist.load_data(path="mnist.npz")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Choose a random sample from the training set
random_index = np.random.randint(0, len(X_train))
random_image = X_train[random_index]
random_label = y_train[random_index]

# Plot the image
plt.imshow(random_image, cmap="gray")
plt.title(f"Label: {random_label}")
plt.axis("off")  # Turn off axes for better visualization
plt.show()


