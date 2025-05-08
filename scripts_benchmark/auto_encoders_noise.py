import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from linear_module import MSELoss
from sequence import Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder, BinaryCrossEntropy

"""
Permet de tester la capacité d'un AE pour débruiter une image
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

def show_imgs(imgs, imgs_noise, labels, networks, net_labels, loss_fn):
    """
    Affiche les images et leurs reconstructions (5 images et labels associés)
    """
    plt.figure(figsize=(12, 5))

    for i in range(len(imgs)):
        print("Image", i)

        # Charger les images
        img1 = imgs[i].reshape(28, 28)
        img2 = imgs_noise[i].reshape(28, 28)


        # Afficher la première image
        plt.subplot(len(imgs), len(networks) + 2, (len(networks) + 2) * i + 1)
        plt.imshow(img1)
        plt.gray()
        plt.title(labels[i] + " initial")
        plt.axis('off') 

        # Afficher l'image bruitée
        plt.subplot(len(imgs), len(networks) + 2, (len(networks) + 2) * i + 2)
        plt.imshow(img2)
        plt.gray()
        plt.title(labels[i] + " bruitée")
        plt.axis('off') 

        for j in range(len(networks)):
            img = networks[j].forward(np.array([imgs_noise[i]])).reshape(28, 28)
            l = loss_fn.forward(img1, img).round(3)

            # Afficher la deuxième image
            plt.subplot(len(imgs), len(networks) + 2, (len(networks) + 2) * i + j + 3)
            plt.imshow(img)
            plt.gray()
            plt.title(f"{net_labels[j]} - L = {l}")
            plt.axis('off')

        plt.tight_layout()

    plt.show()

# Chargement des données et des networks
X, Y = load_data()

noise_intensity = 0.6

# Sélection des images
imgs = [X[np.where(Y == i)[0][0]] for i in range(10)]
imgs_noise = [np.array([min(1.0, max(0.0, px + (np.random.rand() * 2 - 1) * noise_intensity)) for px in img]) for img in imgs]

labels = [str(i) for i in range(10)]


def test_noise():
    # Chargement des réseaux
    network1 = load_network(f"auto_MSELoss_32_16_4_20")
    network2 = load_network(f"auto_BinaryCrossEntropy_32_16_4_50")

    networks = [network1, network2]
    net_labels = ["MSE", "BCE"]

    # Affichage
    show_imgs(imgs[:5], imgs_noise[:5], labels[:5], networks, net_labels, MSELoss()) # Affichage 5 premiers
    show_imgs(imgs[5:], imgs_noise[5:], labels[5:], networks, net_labels, MSELoss()) # Affichage 5 suivants


"""Affichages"""
test_noise()