#DATOS

#importar del python-mnist
from mnist import MNIST

#importar el dataset
mndata = MNIST('MNIST_data')

datos, labels = mndata.load_training()

from matplotlib import pyplot as plt

import numpy as np

plt.gray()

for i in range(25):
    plt.subplot(5,5,i+1)
    
    d_image = datos[i]
    d_image = np.array(d_image, dtype='float')
    
    pixels = d_image.reshape((28,28))
    
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')
    
plt.show()