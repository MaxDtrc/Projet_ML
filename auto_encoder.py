import numpy as np
from linear_module import Linear
from non_linear_module import TanH, Sigmoide
from neural_network import Loss

class AutoEncoder():
    """
    Classe permettant de construire un auto-encoder
    """

    def __init__(self, taille_entree, taille_min, steps):
        """
        Constructeur de la classe AutoEncoder

        Parametres
        taille_entree: dimension des entrées
        taille_min: taille de la plus petite couche
        steps: nombre de couches pour la descente/remontée
        """
        s = (taille_entree - taille_min) // steps

        # Couches de l'encodeur
        self._encoder = []
        for i in range(taille_entree, taille_min + s - 1, - s):
            print("Encodeur - couche", i, "->", i - s, "ajoutée")
            self._encoder.append(Linear(i, i - s))
            self._encoder.append(TanH())

        # Couches du décodeur
        self._decoder = []
        for i in range(taille_min, taille_entree - s + 1, s):
            print("Decodeur - couche", i, "->", i + s, "ajoutée")
            self._decoder.append(Linear(i, i + s))
            self._decoder.append(TanH())

        # Sigmoide en dernier module de la couche decoder
        self._decoder.pop(-1)
        self._decoder.append(Sigmoide())

    
    def forward(self, data):
        """
        Calcule la sortie de la séquence pour les données passées en paramètres

        X: ensemble des données (taille n * dim)
        Sortie: taille n * dimension_sortie
        """
        return self.decode(self.encode(data))
    
    def encode(self, data):
        """
        Calcule l'encodage des données passées en paramètres

        X: ensemble des données (taille n * dim)
        Sortie: taille n * dimension_encodage
        """
        pred = data # Données en entrée

        for mod in self._encoder:
            pred = mod.forward(pred) # Calcul de la sortie de la couche courante

        return pred # On renvoie la sortie
    
    def decode(self, data):
        """
        Decode les données passées en paramètres

        X: ensemble des données (taille n * dimension_encodage)
        Sortie: taille n * dim
        """
        pred = data # Données en entrée

        for mod in self._decoder:
            pred = mod.forward(pred) # Calcul de la sortie de la couche courante

        return pred # On renvoie la sortie
    
    def all_forward(self, data):
        """
        Calcule et renvoie les sorties de chaque couche

        X: ensemble des données (taille n * dim)
        Sortie: taille n * dimension_sortie
        """

        pred = data # Données en entrée
        all_pred = [data]

        for mod in self._encoder:
            pred = mod.forward(pred) # Calcul de la sortie de la couche courante
            all_pred.append(pred)
            
        for mod in self._decoder:
            pred = mod.forward(pred) # Calcul de la sortie de la couche courante
            all_pred.append(pred)

        return all_pred # On renvoie les sorties

    def backward(self, data, delta):
        """
        Met à jour les gradients de chaque couche

        data: donnees en entrée du réseau
        delta: delta de la fonction de coût (ou de la couche suivante)
        """

        preds = self.all_forward(data) # Calcul des entrées de tous les modules
        d = delta # Delta pour la dernière couche de la séquence

        for i in range(len(self._encoder) + len(self._decoder) - 1, -1, -1): # On parcourt les modules en arrière

            if(i >= len(self._encoder)):
                id = i - len(self._encoder)
                self._decoder[id].backward_update_gradient(preds[i], d) # Update du gradient du module
                d = self._decoder[id].backward_delta(preds[i], d) # Calcul du delta
            else:
                self._encoder[i].backward_update_gradient(preds[i], d) # Update du gradient du module
                d = self._encoder[i].backward_delta(preds[i], d) # Calcul du delta

    def zero_grad(self):
        """
        Remet à zero le gradient de chaque module de la séquence
        """
        for mod in self._encoder:
            mod.zero_grad()

        for mod in self._decoder:
            mod.zero_grad()
        
    def update_parameters(self, gradient_step=0.001):
        """
        Met à jour les paramètres de tous les modules
        """
        for mod in self._encoder:
            mod.update_parameters(gradient_step)

        for mod in self._decoder:
            mod.update_parameters(gradient_step)

if __name__ == "__main__":
    ae = AutoEncoder(100, 10, 1)