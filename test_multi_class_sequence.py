import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from multi_class import CrossEntropyWithLogSoftmax
from linear_module import Linear
from non_linear_module import TanH
from sequence import Sequentiel, Optim
import matplotlib.pyplot as plt


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Normalisation des pixel entre [0, 1]
X /= 255.0

# Conversion des labels en one hot encoding
encoder = OneHotEncoder(sparse_output=False)
Y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# L'architecture du modèle
input_size = 784
hidden_size = 128
output_size = 10

modules = [
    Linear(input_size, hidden_size), # 1 ère couche linéaire
    TanH(), # fonction d'activation TanH
    Linear(hidden_size, output_size) # 2 ème couche linéaire
]


# Utilisation de la cross-entropie comme fonction de coût
loss_fn = CrossEntropyWithLogSoftmax()

# Paramètre pour la descente de gradient en mini-batch
learning_rate = 0.01
num_epochs = 10
batch_size = 32

network = Sequentiel(modules)
optim = Optim(network, loss_fn, learning_rate)

# Apprentissage
acc = optim.SGD(X_train, Y_train, batch_size, num_epochs, X_test, Y_test, log = True)


plt.plot(np.arange(len(acc)), acc)
plt.xlabel("Itérations")
plt.ylabel("Accuracy")
plt.title("Evolution de l'accuracy au cours de l'apprentissage")
plt.show()