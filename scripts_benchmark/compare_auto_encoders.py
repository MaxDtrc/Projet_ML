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

def show_imgs(imgs, labels, networks, net_labels, loss_fn):
    """
    Affiche les images et leurs reconstructions (5 images et labels associés)
    """
    plt.figure(figsize=(12, 5))

    for i in range(len(imgs)):
        print("Image", i)

        # Charger les images
        img1 = imgs[i].reshape(28, 28)

        # Afficher la première image
        plt.subplot(len(imgs), len(networks) + 1, (len(networks) + 1) * i + 1)
        plt.imshow(img1)
        plt.gray()
        plt.title(labels[i] + " initial")
        plt.axis('off')  # pour cacher les axes

        for j in range(len(networks)):
            img = networks[j].forward(np.array([imgs[i]])).reshape(28, 28)
            l = loss_fn.forward(img1, img).round(3)

            # Afficher la deuxième image
            plt.subplot(len(imgs), len(networks) + 1, (len(networks) + 1) * i + j + 2)
            plt.imshow(img)
            plt.gray()
            plt.title(f"{net_labels[j]} - L = {l}")
            plt.axis('off')

        plt.tight_layout()

    plt.show()

# Chargement des données et des networks
X, Y = load_data()

# Sélection des images
imgs = [X[np.where(Y == i)[0][0]] for i in range(10)]
labels = [str(i) for i in range(10)]


def compare_mse_layers():
    # Fonction de coût
    loss_fn = MSELoss() 

    # Chargement des réseaux
    network1 = load_network(f"auto_{loss_fn.__class__.__name__}_32_16_1_50")
    network2 = load_network(f"auto_{loss_fn.__class__.__name__}_32_16_2_50")
    network3 = load_network(f"auto_{loss_fn.__class__.__name__}_32_16_4_50")

    networks = [network1, network2, network3]
    net_labels = ["4 couches", "8 couches", "16 couches"]

    # Affichage
    show_imgs(imgs[:5], labels[:5], networks, net_labels, loss_fn) # Affichage 5 premiers
    show_imgs(imgs[5:], labels[5:], networks, net_labels, loss_fn) # Affichage 5 suivants

def compare_mse_neurones():
    # Fonction de coût
    loss_fn = MSELoss() 

    # Chargement des réseaux
    network1 = load_network(f"auto_{loss_fn.__class__.__name__}_32_16_2_50")
    network2 = load_network(f"auto_{loss_fn.__class__.__name__}_32_32_2_50")

    networks = [network1, network2]
    net_labels = ["16 neurones", "32 neurones"]

    # Affichage
    show_imgs(imgs[:5], labels[:5], networks, net_labels, loss_fn) # Affichage 5 premiers
    show_imgs(imgs[5:], labels[5:], networks, net_labels, loss_fn) # Affichage 5 suivants

def compare_bce_batch():
    # Fonction de coût
    loss_fn = BinaryCrossEntropy() 

    # Chargement des réseaux
    network1 = load_network(f"auto_{loss_fn.__class__.__name__}_32_16_2_20")
    network2 = load_network(f"auto_{loss_fn.__class__.__name__}_64_16_2_40")

    networks = [network1, network2]
    net_labels = ["Batch Size = 32", "Batch Size = 64"]

    # Affichage
    show_imgs(imgs[:5], labels[:5], networks, net_labels, loss_fn) # Affichage 5 premiers
    show_imgs(imgs[5:], labels[5:], networks, net_labels, loss_fn) # Affichage 5 suivants



def compare_mse_bce_10():
    # Chargement des réseaux
    network1 = load_network(f"auto_MSELoss_32_16_2_20")
    network2 = load_network(f"auto_BinaryCrossEntropy_32_16_2_50")

    networks = [network1, network2]
    net_labels = ["MSE", "BCE"]

    # Affichage
    show_imgs(imgs[:5], labels[:5], networks, net_labels, MSELoss()) # Affichage 5 premiers
    show_imgs(imgs[5:], labels[5:], networks, net_labels, MSELoss()) # Affichage 5 suivants


"""Affichages"""
#compare_mse_layers()
#compare_mse_neurones()
#compare_mse_bce_10()
compare_bce_batch()