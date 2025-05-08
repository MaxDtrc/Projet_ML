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
Permet d'entrainer plusieurs réseaux avec des configurations différentes et de les sauvegarder
dans des fichiers
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

# Conversion des labels en one hot encoding
Y_onehot = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

print("Données d'entrainement chargées et traitées")

# L'architecture du modèle
input_size = 784

learning_rates = [0.01, 0.1]
loss_functions = [MSELoss(), BinaryCrossEntropy()]
max_epochs = [1, 5]
batch_sizes = [32, 64]
min_size_lst = [16, 32]
steps_list = [1, 2, 4]

i = 0
for epochs, learning_rate, loss_fn in zip(max_epochs, learning_rates, loss_functions):
    for batch_size in batch_sizes:
        for min_size in min_size_lst:
            for steps in steps_list:
                # Creation du reseau
                network = AutoEncoder(input_size, min_size, steps)

                # Creation de l'optimiseur
                optim = Optim(network, loss_fn, learning_rate)

                # Apprentissage
                for j in range(epochs):
                    l, _ = optim.SGD(X_train, X_train, batch_size, 10, log = True)

                    # Sauvegarde du réseau
                    file_name = f"auto_{loss_fn.__class__.__name__}_{batch_size}_{min_size}_{steps}_{(j+1) * 10}"
                    print(file_name, "sauvegardé")

                    i += 1
                    network.save(file_name)

print(f"Entrainement Terminé ({i})")