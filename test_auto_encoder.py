import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from linear_module import MSELoss
from sequence import Optim
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder, BinaryCrossEntropy
from skimage.transform import resize

def downscale_images(X, new_size=(16, 16)):
    """
    Fonction permettant de réduire les dimensions des images
    """
    return np.array([resize(img, new_size, mode='reflect', anti_aliasing=True) for img in X])

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32)
Y = mnist.target.astype(int)

indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices] # On ne garde que 20 000 images pour accelerer l'entrainement

# Normalisation des pixel entre [0, 1]
X /= 255.0

# Downscale des images
#X = downscale_images(X.reshape(X.shape[0], 28, 28)).reshape(X.shape[0], 256)

# Conversion des labels en one hot encoding
Y_onehot = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

print("Données d'entrainement chargées et traitées")

# L'architecture du modèle
input_size = 784
min_size = 32
steps = 4

# Creation du reseau
network = AutoEncoder(input_size, min_size, steps)

# Creation de l'optimiseur
loss_fn = BinaryCrossEntropy()
learning_rate = 0.1

network.load("auto_encoder_bce_3.txt")

optim = Optim(network, loss_fn, learning_rate)

# Paramètre pour la descente de gradient en mini-batch
num_epochs = 5
batch_size = 32

# Apprentissage
l, _ = optim.SGD(X_train, X_train, batch_size, num_epochs, log = True)

# Sauvegarde du réseau
network.save("auto_encoder_bce_4.txt")

# Chargement du réseau
#network.load("auto_encoder_mse.txt")

# Premier chiffre de base

for i in range(20):
    plt.imshow(X_train[i].reshape(28, 28))
    plt.gray()
    plt.show()

    plt.imshow(network.forward(np.array([X_train[i]])).reshape(28, 28))
    plt.gray()
    plt.show()

# Affichage de l'évolution de l'accuracy sur les données de test :
#plt.plot(np.arange(len(l)), l)
#plt.show()