from linear_module import MSELoss, Linear
from non_linear_module import TanH, Sigmoide
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Test des modules linéaires avec 2 couches (sans fonction d'activation donc c'est un juste perceptron déguisé en vrai)

loss = MSELoss()
c1 = Linear(2, 5) # Couche 1, 5 neurones
c2 = Linear(5, 1) # Couche 2, fusion des neurones
acti_tanh = TanH()
acti_sigm = Sigmoide()

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

# Boucle màj 
accuracy = []
for i in range(2000):
    print("Itération", i)

    couche1 = c1.forward(x_train) # Sortie de la couche 1
    activation1 = acti_tanh.forward(couche1) # Fonction d'activation TanH
    couche2 = c2.forward(activation1) # Sortie de la couche 2
    pred = acti_sigm.forward(couche2) # Prédiction
    grad_loss = loss.backward(y_train, pred) # Calcul du gradient de la loss
    
    delta_sigm = acti_sigm.backward_delta(couche2, grad_loss) # Calcul du delta pour la sortie
    c2.backward_update_gradient(activation1, delta_sigm)  # Màj du gradient de la couche 2
    delta_c2 = c2.backward_delta(activation1, delta_sigm)  # Calcul du delta des entrées de la couche 2
    delta_tanh = acti_tanh.backward_delta(couche1, delta_c2)  # Calcul du gradient de la fonction d'activation
    c1.backward_update_gradient(x_train, delta_tanh)  # Màj du gradient de la couche 1
    delta_c1 = c1.backward_delta(x_train, delta_tanh)  # Calcul du delta des entrées de la couche 1 

    # Mise à jour des paramètres
    c2.update_parameters(0.001)
    c1.update_parameters(0.001)
    c2.zero_grad()
    c1.zero_grad()

    # Calcul des performances
    pred_train = np.sign(c2.forward(c1.forward(x_test)))
    accuracy.append(np.mean(pred_train == y_test))

# Affichage de l'évolution de l'accuracy sur les données de test :
plt.plot(np.arange(len(accuracy)), accuracy)
plt.show()