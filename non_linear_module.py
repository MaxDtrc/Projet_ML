from neural_network import Loss, Module
import numpy as np

class TanH(Module):
    """
    Classe représentant une fonction d'activation Tangente Hyperbolique.
    """
    def __init__(self):
        """
        Constructeur de la classe TanH
        """
        # Appel du constructeur de la classe mère, aucun paramètre supplémentaire
        super().__init__()  

    def forward(self, data):
        """
        Calcule la sortie de la couche pour chacune des valeurs passées en paramètres
        en appliquant la fonction tanH.

        X : ensemble des données (taille n * dim)
        Sortie : taille n * output_size
        """
        self.output = np.tanh(data)
        return self.output

    def backward_delta(self, input, delta):
        """
        Calcule le gradient des entrées en fonction des entrées
        de la couche suivante

        input : entrées (n'entrent pas en compte dans le calcul)
        delta: gradient des entrées de la couche suivante
        """

        return delta * (1 - self.output ** 2)

    def backward_update_gradient(self, input, delta):
        pass  # Pas de paramètres à mettre à jour

    def update_parameters(self, gradient_step):
        pass  # Pas de mise à jour nécessaire
    
    def zero_grad(self):
        pass # Pas de gradient à mettre à jour

    def __str__(self):
        return "TanH"

class Sigmoide(Module):
    """
    Classe représentant une fonction d'activation Sigmoïde
    """

    def __init__(self):
        """
        Constructeur de la classe Sigmoide
        """
        # Appel du constructeur de la classe mère
        super().__init__()  
    
    def forward(self, data):
        """
        Calcule la sortie de la couche pour chacune des valeurs passées en paramètres
        en appliquant la fonction sigmoide.

        X : ensemble des données (taille n * dim)
        Sortie : taille n * output_size
        """
        self.output = 1 / (1 + np.exp(-data))
        return self.output

    def backward_delta(self, input, delta):
        """
        Calcule le gradient des entrées en fonction des entrées
        de la couche suivante, en utilisant la dérivée de la fonction
        sigmoide : σ(x) * (1 - σ(x))

        input : entrées (n'entrent pas en compte dans le calcul)
        delta: gradient des entrées de la couche suivante
        """

        return delta * self.output * (1 - self.output)

    def backward_update_gradient(self, input, delta):
        pass  # Pas de paramètres à mettre à jour

    def update_parameters(self, gradient_step):
        pass  # Pas de mise à jour nécessaire

    def zero_grad(self):
        pass # Pas de gradient à mettre à jour
        
    def __str__(self):
        return "Sigmoide"