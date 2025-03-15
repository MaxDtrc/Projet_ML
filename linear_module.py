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
        self.input_size = input_size
        self.output_size = output_size
        self._parameters = np.ones(input_size)

    def zero_grad(self):
        pass
    
    def forward(self, X):
        assert(X.shape[1] == self.input_size)
        pass
    
    def update_parameters(self, gradient_step=0.001):
        pass
    
    def backward_update_gradient(self, input, delta):
        pass
    
    def backward_delta(self, input, delta):
        pass

if __name__ == "__main__":
    """
    Test de la MSE Loss
    """
    #Données artificielles
    test = np.array([[0.1, 2, 4],
            [0.2, 3, 6]])
    
    #Données prédites artificielles
    test_predict = np.array([[0.15, 2.1, 4.2],
                    [0.5, 6, 1]])
    
    l = MSELoss()
    print(l.forward(test, test_predict)) #Calcul de l'erreur