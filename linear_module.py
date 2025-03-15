from neural_network import Loss, Module
import numpy as np

class MSELoss(Loss):
    def forward(self, y, y_hat):
        """
        Renvoie la MSE des données passées en paramètres
        y: ensemble des étiquettes des données (n * d)
        y_hat: ensemble des étiquettes prédites (n * d)
        """
        assert(y.shape == y_hat.shape) # Vérification des dimensions

        n, d = y.shape
        return 1/d * np.sum(np.square(y - y_hat), axis=1)

    def backward(self, y, y_hat):
        """
        Renvoie le gradient du coût par rapport aux données prédites y_hat

        y: ensemble des étiquettes des données (n)
        y_hat: ensemble des étiquettes prédites (n)
        """
        assert(y.shape == y_hat.shape) # Vérification des dimensions

        n, d = y.shape
        return 2/d * (y_hat - y)
        
class Linear(Module):
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._parameters = np.ones(shape=(output_size, input_size)) # Poids de pour chaque dimension de l'entrée
        self._gradient = np.zeros(self._parameters.shape) # Mise à 0 du gradient

    def zero_grad(self):
        """
        Remet à 0 le gradient
        """
        self._gradient = np.zeros(self._parameters.shape)
    
    def forward(self, X):
        """
        Calcule la sortie de la couche pour chacune des valeurs passées en paramètres, en fonction de la matrice des poids

        X : ensemble des données (taille n * dim)
        Sortie : taille n * output_size
        """
        assert(X.shape[1] == self._input_size) # Vérification dimension des entrées
        out = np.dot(X, self._parameters.T) # Calcul des valeurs de sortie (sans biais pour l'instant)
        assert(out.shape == (X.shape[0], self._output_size)) # Vérification dimension des sorties
        return out
    
    def update_parameters(self, gradient_step=0.001):
        """
        Met à jour les paramètres en fonction du gradient accumulé et du pas de gradient
        """
        super().update_parameters(gradient_step)
    
    def backward_update_gradient(self, input, delta):
        """
        Calcule le gradient des paramètres en fonction d'une entrée et des delta de la couche suivante.
        Ici, il s'agit d'une couche linéaire. Donc, d(w_ij) = delta_j * x_i

        input : une entrée (taille d)
        delta : dérivées des entrées de la couche suivante (taille output_size)
        """
        assert(input.shape == (self._input_size,)) # Vérification taille de l'entrée
        assert(delta.shape == (self._output_size,)) # Vérification taille du tableau des delta

        grad = np.outer(delta, input) # Calcul du gradient pour chaque paramètre

        assert(grad.shape == self._gradient.shape) # Vérification taille du gradient
        self._gradient += grad # Accumulation du gradient

    
    def backward_delta(self, input, delta):
        """
        Calcule le gradient des entrées par rapport à une entrée et aux dérivées des entrées de la couche suivante.
        Ici, il s'agit d'une couche linéaire. Donc, d(z_i) = somme_sur_j(x_i * z_j * w_ij)

        input : une entrée (taille d)
        delta : dérivées des entrées de la couche suivante (taille output_size)
        """
        assert(input.shape == (self._input_size,)) # Vérification taille de l'entrée
        assert(delta.shape == (self._output_size,)) # Vérification taille du tableau des delta

        xi_zj = np.outer(delta, input)
        assert(xi_zj.shape == self._parameters.shape)
        z_i = np.sum(np.multiply(self._parameters, xi_zj), axis = 0)

        assert(z_i.shape[0] == self._input_size)
        return z_i
        

if __name__ == "__main__":
    """
    Test de la MSE Loss
    """
    # Données artificielles
    test = np.array([[0.1, 2, 4],
            [0.2, 3, 6]])
    
    # Données prédites artificielles
    test_predict = np.array([[0.15, 2.1, 4.2],
                    [0.5, 6, 1]])
    
    # Calcul de la loss
    l = MSELoss()
    print("Loss =", l.forward(test, test_predict)) #Calcul de l'erreur

    # Calcul de la sortie d'une couche linéaire
    lin = Linear(3, 5)
    res = lin.forward(test)
    print("\nSortie couche =", res)

    # Backward
    for x in test:
        lin.backward_update_gradient(x, np.ones(5))
        #print(lin._gradient)

        print(lin.backward_delta(x, np.ones(5)))