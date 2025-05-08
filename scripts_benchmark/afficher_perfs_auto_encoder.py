import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from linear_module import MSELoss
from sequence import Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder, BinaryCrossEntropy
from skimage.transform import resize

"""
Permet d'afficher au hasard 10 chiffres après passage dans l'auto-encoder (chargé depuis un fichier)
"""

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32)
Y = mnist.target.astype(int)

indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices]


# Normalisation des pixel entre [0, 1]
X /= 255.0

# Creation du reseau
network = AutoEncoder()

# Chargement du réseau
network.load("auto_BinaryCrossEntropy_32_32_4_50")

plt.figure(figsize=(10, 5))
for i in range(10):
    print("Image", i)

    # Charger les images
    img1 = X[i].reshape(28, 28)
    img2 = network.forward(np.array([X[i]])).reshape(28, 28)

    # Afficher la première image
    plt.subplot(5, 4, 2*i + 1)
    plt.imshow(img1)
    plt.gray()
    plt.title(str(Y[i]) + " initial")
    plt.axis('off')  # pour cacher les axes

    # Afficher la deuxième image
    plt.subplot(5, 4, 2*i + 2)
    plt.imshow(img2)
    plt.gray()
    plt.title(str(Y[i]) + " après AE")
    plt.axis('off')

    plt.tight_layout()


plt.show()
