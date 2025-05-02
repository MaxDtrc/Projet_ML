import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from multi_class import CrossEntropyWithLogSoftmax
from linear_module import Linear
from non_linear_module import TanH
from sequence import Sequentiel, Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder, BinaryCrossEntropyWithClip

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Normalisation des pixel entre [0, 1]
X /= 255.0

# Conversion des labels en one hot encoding
Y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# L'architecture du modèle
input_size = 784
min_size = 184
steps = 10

# Creation du reseau
network = AutoEncoder(input_size, min_size, steps)

# Creation de l'optimiseur
loss_fn = BinaryCrossEntropyWithClip()
learning_rate = 0.01

optim = Optim(network, loss_fn, learning_rate)

# Boucle màj 
accuracy = []
for i in range(2000):
    print("Itération", i)

    optim.step(X_train, X_train) # Itération de la descente

    # Calcul des performances
    pred_test = np.sign(optim._net.forward(x_test))
    accuracy.append(np.mean(pred_test == y_test))
    #print(accuracy[i])

# Affichage de l'évolution de l'accuracy sur les données de test :
plt.plot(np.arange(len(accuracy)), accuracy)
plt.show()