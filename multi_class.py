from neural_network import Loss, Module
import numpy as np

class CrossEntropyWithLogSoftmax(Loss):
    """
    Classe représentant un module d'une fonction de coût cross-entropie avec Softmax passé au log
    """

    def forward(self, y, y_hat):
        """
        Renvoie la cross-entropie des données passées en paramètres

        Parametres
        y: ensemble des étiquettes des données (n * d)
        y_hat: ensemble des étiquettes prédites (n * d)
        """
        assert(y.shape == y_hat.shape) # Vérification des dimensions

        # Stabilité numérique avec log-sum-exp trick
        max_logits = np.max(y_hat, axis=1, keepdims=True)
        logsumexp = max_logits + np.log(np.sum(np.exp(y_hat - max_logits), axis=1, keepdims=True)) 
        correct_class_logits = np.sum(y * y_hat, axis=1, keepdims=True)

        loss = -correct_class_logits + logsumexp
        return np.mean(loss)

    def backward(self, y, y_hat):
        """
        Renvoie le gradient du coût par rapport aux données prédites y_hat

        Parametres
        y: ensemble des étiquettes des données (n)
        y_hat: ensemble des étiquettes prédites (n)
        """
        assert(y.shape == y_hat.shape) # Vérification des dimensions

        exp_logits = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        grad = (softmax - y) / y.shape[0]  # division par batch_size
        return grad




# On créer un Module Softmax, mais il n'est pas réellement utile   
class Softmax(Module):
    """
    Classe représentant une fonction d'activation Softmax.
    """
    def __init__(self):
        """
        Constructeur de la classe Softmax
        """
        # Appel du constructeur de la classe mère, aucun paramètre supplémentaire
        super().__init__()  

    def forward(self, data):
        """
        Calcule la sortie de la couche pour chacune des valeurs passées en paramètres
        en appliquant la fonction Softmax.

        Parametres
        X : ensemble des données (taille n * dim)
        Sortie : taille n * output_size
        """
        e_x = np.exp(data - np.max(data, axis=-1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.output

    def backward_delta(self, input, delta):
        """
        Calcule le gradient des entrées en fonction des entrées
        de la couche suivante

        Parametres
        input : entrées (n'entrent pas en compte dans le calcul)
        delta: gradient des entrées de la couche suivante
        """

        return delta * (np.diag(self.output) - np.outer(self.output, self.output))

    def backward_update_gradient(self, input, delta):
        pass  # Pas de paramètres à mettre à jour

    def update_parameters(self, gradient_step):
        pass  # Pas de mise à jour nécessaire
    
    def zero_grad(self):
        pass # Pas de gradient à mettre à jour
