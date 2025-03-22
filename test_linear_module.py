from linear_module import MSELoss, Linear
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Test des modules linéaires avec 2 couches (sans fonction d'activation donc c'est un juste perceptron déguisé en vrai)

loss = MSELoss()
c1 = Linear(2, 5) # Couche 1, 5 neurones
c2 = Linear(5, 1) # Couche 2, fusion des neurones

# Génération de données artificielles
X, Y = make_classification(n_classes=2, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, n_samples=1000)

Y = Y * 2 - 1
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Reshape des données
#x_train = x_train.reshape(-1, 2)
#x_test = x_test.reshape(-1, 2)
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

    pred = c2.forward(c1.forward(x_train)) # Prediction
    grad_loss = loss.backward(y_train, pred) # Calcul du gradient de la loss
    
    for x, delta in zip(x_train, grad_loss):
        x_data = np.array([x])

        pred_c1 = c1.forward(x_data)
        c2.backward_update_gradient(pred_c1[0], delta) # Màj du gradient de la couche 2
        c2_grad = c2.backward_delta(pred_c1[0], delta) # Calcul du gradient des entrées de la couche 2
        c1.backward_update_gradient(x, c2_grad) # Màj du gradient de la couche 1

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