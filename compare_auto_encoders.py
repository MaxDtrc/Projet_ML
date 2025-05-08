import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from linear_module import MSELoss
from sequence import Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder, BinaryCrossEntropy

"""
Permet de comparer les différentes configurations selon plusieurs critères
"""

def load_data():
    """
    Charge les données du dataset MNIST en renvoie les données traitées
    """

    # Charge les données
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X = mnist.data.astype(np.float32)
    Y = mnist.target.astype(int)

    # Permutation aléatoire
    #indices = np.random.permutation(len(X))
    #X = X[indices]
    #Y = Y[indices]


    # Normalisation des pixel entre [0, 1]
    X /= 255.0

    return X, Y

def load_network(name):
    """
    Permet de charger l'auto-encoder passé en argument depuis un fichier
    """
    network = AutoEncoder() # Creation du reseau
    network.load(name) # Chargement du réseau
    return network 

def show_imgs(imgs, labels):
    """
    Affiche les images et leurs reconstructions (5 images et labels associés)
    """
    plt.figure(figsize=(12, 5))

    for i in range(5):
        print("Image", i)

        # Charger les images
        img1 = imgs[i].reshape(28, 28)
        img2 = network1.forward(np.array([imgs[i]])).reshape(28, 28)
        img3 = network2.forward(np.array([imgs[i]])).reshape(28, 28)
        img4 = network3.forward(np.array([imgs[i]])).reshape(28, 28)

        loss_fn = MSELoss()
        l1 = loss_fn.forward(img1, img2).round(2)
        l2 = loss_fn.forward(img1, img3).round(2)
        l3 = loss_fn.forward(img1, img4).round(2)

        # Afficher la première image
        plt.subplot(5, 4, 4*i + 1)
        plt.imshow(img1)
        plt.gray()
        plt.title(labels[i] + " initial")
        plt.axis('off')  # pour cacher les axes

        # Afficher la deuxième image
        plt.subplot(5, 4, 4*i + 2)
        plt.imshow(img2)
        plt.gray()
        plt.title(f"2 couches - L = {l1}")
        plt.axis('off')

        # Afficher la deuxième image
        plt.subplot(5, 4, 4*i + 3)
        plt.imshow(img3)
        plt.gray()
        plt.title(f"4 couches - L = {l2}")
        plt.axis('off')

        # Afficher la deuxième image
        plt.subplot(5, 4, 4*i + 4)
        plt.imshow(img4)
        plt.gray()
        plt.title(f"8 couches - L = {l3}")
        plt.axis('off')

        plt.tight_layout()

    plt.show()

# Chargement des données et des networks
X, Y = load_data()

network1 = load_network("auto_MSELoss_32_16_1_10")
network2 = load_network("auto_MSELoss_32_16_2_10")
network3 = load_network("auto_MSELoss_32_16_4_10")

# Sélection des images
imgs = [X[np.where(Y == i)[0][0]] for i in range(10)]
labels = [i for i in range(10)]

# Affichage
show_imgs(imgs[:5], labels[:5]) # Affichage 5 premiers
show_imgs(imgs[5:], labels[5:]) # Affichage 5 suivants

