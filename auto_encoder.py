import os
import numpy as np
from linear_module import Linear
from non_linear_module import TanH, Sigmoide
from neural_network import Loss
from sequence import Sequentiel

class BinaryCrossEntropy(Loss):
    """
    Classe représentant un module d'une fonction de coût Cross Entropie Binaire
    """

    def forward(self, y, y_hat):
        """
        Renvoie le coût des données passées en paramètres
        y: ensemble des étiquettes des données (n * d)
        y_hat: ensemble des étiquettes prédites (n * d)
        """
        assert y.shape == y_hat.shape

        # Clipping pour éviter log(0)
        eps = 1e-7
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)

        # Calcul de la loss
        loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return loss
    

    def backward(self, y, y_hat):
        """
        Renvoie le gradient du coût par rapport aux données prédites y_hat

        y: ensemble des étiquettes des données (n)
        y_hat: ensemble des étiquettes prédites (n)
        """
        assert y.shape == y_hat.shape

        # Clipping pour éviter division par zéro
        eps = 1e-7
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)

        # Calcul du gradient
        grad = - (y / y_hat_clipped - (1 - y) / (1 - y_hat_clipped)) / y.size
        return grad

class AutoEncoder():
    """
    Classe permettant de construire un auto-encoder
    """

    def __init__(self, taille_entree = 256, taille_min = 16, steps = 2):
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

    def save(self, name):
        """
        Sauvegarde le réseau de neurone dans un fichier texte
        """
        with open(os.path.join("networks", name), "w") as file:
            for mod in self._encoder:
                if hasattr(mod, "_input_size") and hasattr(mod, "_input_size"):
                    # Module avec paramètres
                    file.write("e " + mod.__class__.__name__ + " " + str(mod._input_size) + " " + str(mod._output_size) + '\n')

                    file.write(str(mod._parameters.tolist()) + '\n')
                    file.write(str(mod._bias.tolist()) + '\n')

                else:
                    # Module sans paramètres
                    file.write("e " + mod.__class__.__name__ + '\n')
                
                file.write("\n")

            for mod in self._decoder:
                if hasattr(mod, "_input_size") and hasattr(mod, "_input_size"):
                    # Module avec paramètres
                    file.write("d " + mod.__class__.__name__ + " " + str(mod._input_size) + " " + str(mod._output_size) + '\n')

                    file.write(str(mod._parameters.tolist()) + '\n')
                    file.write(str(mod._bias.tolist()) + '\n')

                else:
                    # Module sans paramètres
                    file.write("d " + mod.__class__.__name__ + '\n')
                
                file.write("\n") 

        print("Réseau sauvegardé")

    def load(self, name):
        """
        Charge un réseau de neurones depuis un fichier texte
        """
        self._encoder = []
        self._decoder = []

        with open(os.path.join("networks", name), "r") as file:
            # Lecture du fichier
            lines = file.readlines()
            i = 0

            # Parcours des lignes
            while i < len(lines):
                l = lines[i]
                ls = l.replace('\n', '').split(" ")

                if len(ls) == 4:
                    # Modules avec paramètres
                    mod = eval(ls[1])(int(ls[2]), int(ls[3])) # Création du module

                    # Changement de ligne
                    i += 1
                    l = lines[i].replace('\n', '')

                    mod._parameters = np.array(eval(l)) # Charement des paramètres

                    # Changement de ligne
                    i += 1
                    l = lines[i].replace('\n', '')

                    mod._bias = np.array(eval(l)) # Chargement du biais
                    
                    # Changement de ligne
                    i += 1
                    l = lines[i]

                    assert l == '\n' # On vérifie qu'on est bien à la fin du module

                if len(ls) == 2:
                    # Module sans paramètres
                    mod = eval(ls[1])() # Création du module

                    # Changement de ligne
                    i += 1
                    l = lines[i]

                    assert l == '\n' # On vérifie qu'on est bien à la fin du module

                # Ajout du module
                if ls[0] == "e":
                    self._encoder.append(mod)
                elif ls[0] == 'd':
                    self._decoder.append(mod)

                i += 1 # Changement de ligne

        print("Réseau chargé")

if __name__ == "__main__":
    ae = AutoEncoder(100, 10, 1)