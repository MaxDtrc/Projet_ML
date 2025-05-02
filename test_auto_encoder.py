import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from multi_class import CrossEntropyWithLogSoftmax
from linear_module import Linear, MSELoss
from non_linear_module import TanH
from sequence import Sequentiel, Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32)
Y = mnist.target.astype(int)

indices = np.random.permutation(len(X))
X = X[indices][:100]
Y = Y[indices][:100]

# Normalisation des pixel entre [0, 1]
X /= 255.0

# Conversion des labels en one hot encoding
Y_onehot = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# L'architecture du modèle
input_size = 784
min_size = 184
steps = 2

# Creation du reseau
network = AutoEncoder(input_size, min_size, steps)

# Creation de l'optimiseur
loss_fn = MSELoss()
learning_rate = 0.001

optim = Optim(network, loss_fn, learning_rate)

# Boucle màj 
l = []
for i in range(3000):
    print("Itération", i)
    optim.step(X_train, X_train) # Itération de la descente

    # Calcul de la loss sur les données d'entrainement
    l.append(loss_fn.forward(network.forward(X_test), X_test))

# Affichage de l'évolution de l'accuracy sur les données de test :
plt.plot(np.arange(len(l)), l)
plt.show()