import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from multi_class import CrossEntropyWithLogSoftmax
from linear_module import Linear
from non_linear_module import TanH

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Normalisation des pixel entre [0, 1]
X /= 255.0

# Conversion des labels en one hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Séparation de donnée en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# L'architecture du modèle
input_size = 784
hidden_size = 128
output_size = 10

model = [
    Linear(input_size, hidden_size), # 1 ère couche linéaire
    TanH(), # fonction d'activation TanH
    Linear(hidden_size, output_size) # 2 ème couche linéaire
]

# Utilisation de la cross-entropie comme fonction de coût
loss_fn = CrossEntropyWithLogSoftmax()

# Paramètre pour la descente de gradient en mini-batch
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Boucle d'apprentissage
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # mélange des données d'entrainement
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward 
        activations = [X_batch]
        for layer in model:
            activations.append(layer.forward(activations[-1]))
        y_hat = activations[-1]
        
        #Loss
        loss = loss_fn.forward(y_batch, y_hat)
        epoch_loss += loss * len(X_batch)
        
        #Accuracy
        preds = np.argmax(y_hat, axis=1)
        labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == labels)
        total += len(X_batch)
        
        #Backward
        grad_loss = loss_fn.backward(y_batch, y_hat)
        delta = grad_loss

        # Propagation des delta à travers les couches
        for layer_idx in reversed(range(len(model))):
            layer = model[layer_idx]
            input_act = activations[layer_idx]
            
            # 1. Mettre à jour les gradients des paramètres AVANT de modifier delta
            if isinstance(layer, Linear):
                layer.backward_update_gradient(input_act, delta)
            
            # 2. Calculer le delta pour la couche précédente
            delta = layer.backward_delta(input_act, delta)
        
        # Mis à jour des param et rénitialisation du gradient
        for layer in model:
            if isinstance(layer, Linear):
                layer.update_parameters(gradient_step=learning_rate)
                layer.zero_grad()
    
    epoch_loss /= len(X_shuffled)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

# Evaluation
activations = [X_test]
for layer in model:
    activations.append(layer.forward(activations[-1]))
y_hat_test = activations[-1]

test_preds = np.argmax(y_hat_test, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_acc = np.mean(test_preds == test_labels)
print(f"\nTest Accuracy: {test_acc:.4f}")