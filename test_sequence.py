from linear_module import MSELoss, Linear
from non_linear_module import TanH, Sigmoide
from sequence import Sequentiel, Optim
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Test des séquences de modules

# Génération de données artificielles
X, Y = make_classification(n_classes=2, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, n_samples=1000)

Y = Y * 2 - 1
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Reshape des données
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Affichage des points de train
plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)
plt.scatter(x_test[:, 0], x_test[:, 1], marker='x', c=y_test)
plt.show()

# Création du réseau
modules = [Linear(2, 5), TanH(), Linear(5, 1), Sigmoide()]
network = Sequentiel(modules)
optim = Optim(network, MSELoss(), 0.1)

# Boucle màj 
accuracy = []
for i in range(2000):
    #print("Itération", i)

    optim.step(x_train, y_train) # Itération de la descente

    # Calcul des performances
    pred_test = np.sign(optim._net.forward(x_test))

    
    accuracy.append(np.mean(pred_test == y_test))
    print(accuracy[i])

# Affichage de l'évolution de l'accuracy sur les données de test :
plt.plot(np.arange(len(accuracy)), accuracy)
plt.show()