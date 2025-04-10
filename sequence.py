class Sequentiel():
    """
    Classe permettant de construire une série de modules
    """

    def __init__(self, modules):
        """
        Constructeur de la classe Sequentiel

        Parametres
        modules : liste des modules à ajouter
        loss : fonction de coût à utiliser
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

        x: données en entrée
        Y: étiquettes des données
        """

        pred = self._net.forward(X) # Prédiction des étiquettes
        grad_loss = self._loss.backward(Y, pred) # Gradient de la fonction de cout

        self._net.backward(X, grad_loss) # Calcul des gradients pour chaque module de la séquence
        self._net.update_parameters(self._eps) # Mise à jour des paramètres

        self._net.zero_grad() # Remise à zero des gradients 
