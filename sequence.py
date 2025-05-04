import os
import numpy as np

class Sequentiel():
    """
    Classe permettant de construire une série de modules
    """

    def __init__(self, modules):
        """
        Constructeur de la classe Sequentiel

        Parametres
        modules : liste des modules à ajouter
        """
        self._modules = modules

    def forward(self, data):
        """
        Calcule la sortie de la séquence pour les données passées en paramètres

        X: ensemble des données (taille n * dim)
        Sortie: taille n * dimension_sortie
        """
        pred = data # Données en entrée

        for mod in self._modules:
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

        for mod in self._modules:
            pred = mod.forward(pred) # Calcul de la sortie de la couche courante
            all_pred.append(pred)

        return all_pred # On renvoie les sorties

    def backward(self, data, delta):
        """
        Met à jour les gradients de chaque module

        data: donnees en entrée du réseau
        delta: delta de la fonction de coût (ou de la couche suivante)
        """

        preds = self.all_forward(data) # Calcul des entrées de tous les modules
        d = delta # Delta pour la dernière couche de la séquence

        for i in range(len(self._modules) - 1, -1, -1): # On parcourt les modules en arrière
            self._modules[i].backward_update_gradient(preds[i], d) # Update du gradient du module
            d = self._modules[i].backward_delta(preds[i], d) # Calcul du delta

    def zero_grad(self):
        """
        Remet à zero le gradient de chaque module de la séquence
        """
        for mod in self._modules:
            mod.zero_grad()
        
    def update_parameters(self, gradient_step=0.001):
        """
        Met à jour les paramètres de tous les modules
        """
        for mod in self._modules:
            mod.update_parameters(gradient_step)

    def save(self, name):
        """
        Sauvegarde le réseau de neurone dans un fichier texte
        """
        with open(os.path.join("networks", name), "w") as file:
            for mod in self._modules:
                if hasattr(mod, "_input_size") and hasattr(mod, "_input_size"):
                    # Module avec paramètres
                    file.write(mod.__class__.__name__ + " " + str(mod._input_size) + " " + str(mod._output_size) + '\n')

                    file.write(str(mod._parameters.tolist()) + '\n')
                    file.write(str(mod._bias.tolist()) + '\n')

                else:
                    # Module sans paramètres
                    file.write(mod.__class__.__name__ + '\n')
                
                file.write("\n")

        print("Réseau sauvegardé")

    def load(self, name):
        """
        Charge un réseau de neurones depuis un fichier texte
        """
        self._modules = []

        with open(os.path.join("networks", name), "r") as file:
            # Lecture du fichier
            lines = file.readlines()
            i = 0

            # Parcours des lignes
            while i < len(lines):
                l = lines[i]
                ls = l.replace('\n', '').split(" ")

                if len(ls) == 3:
                    # Modules avec paramètres
                    mod = eval(ls[0])(int(ls[1]), int(ls[2])) # Création du module

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

                if len(ls) == 1:
                    # Module sans paramètres
                    mod = eval(ls[0])() # Création du module

                    # Changement de ligne
                    i += 1
                    l = lines[i]

                    assert l == '\n' # On vérifie qu'on est bien à la fin du module

                # Ajout du module
                self._modules.append(mod)

                i += 1 # Changement de ligne

        print("Réseau chargé")

class Optim():
    """
    Classe permettant d'effectuer une descente de gradient sur une séquence de modules
    """

    def __init__(self, net, loss, eps):
        """
        Constructeur de la classe Optim

        Paramètres
        net: séquence des modules
        loss: fonction de coût utilisée pour la descente
        eps: pas pour la descente de gradient
        """
        self._net = net
        self._loss = loss
        self._eps = eps

    def step(self, X, Y):
        """
        Effectue une itération de la descente de gradient

        X: données en entrée
        Y: étiquettes des données
        """

        pred = self._net.forward(X) # Prédiction des étiquettes
        grad_loss = self._loss.backward(Y, pred) # Gradient de la fonction de cout

        self._net.backward(X, grad_loss) # Calcul des gradients pour chaque module de la séquence
        self._net.update_parameters(self._eps) # Mise à jour des paramètres

        self._net.zero_grad() # Remise à zero des gradients 

    def SGD(self, X, Y, batch_size, num_epochs, X_test = None, Y_test = None, log = False):
        """
        Effectue l'apprentissage du réseau pendant un certain nombre d'itérations en mode mini-batch

        X: données en entrées
        Y: étiquettes des données
        batch_size: taille des batch pour le découpage
        num_epochs: nombre d'itérations pour l'apprentissage
        X_test: données de test pour évaluer l'accuracy à chaque itétation
        Y_test: labels de test pour évaluer l'accuracy à chaque itération
        log: affichage des logs
        """
        acc = []
        loss_train = []
        
        print("Apprentissage du réseau ...")
        for epoch in range(num_epochs):
            if log:
                print("Itération", epoch)

            # On mélange les données d'entraînement
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, len(X_shuffled), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]

                # Step sur l'échantillon courant
                self.step(X_batch, Y_batch)

            # On calcule la loss
            loss_train.append(self._loss.forward(self._net.forward(X_shuffled), Y_shuffled))

            if log:
                print("Loss:", loss_train[-1])

            # Test de l'accuracy
            if X_test is not None and Y_test is not None:
                pred = np.argmax(self._net.forward(X_test), axis=1)
                test_labels = np.argmax(Y_test, axis = 1)
                accuracy = np.mean(pred == test_labels)
                acc.append(accuracy)

                if log:
                    print("Accuracy =", accuracy)

        # On retourne le tableau des accuracy si des données de test sont fournies, None sinon
        print("Apprentissage terminé")
        return (loss_train, acc) if X_test is not None and Y_test is not None else (loss_train, None)