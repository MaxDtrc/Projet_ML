

Structure du code : 

### networks ###
    Contient un ensemble d'auto-encoders pré-entraînés
    -> /!\ pour utiliser les réseaux pré-entraînés, dé-zipper l'archive "networks.zip" dans le dossier networks

### scripts_benchmark ###
    Contient un ensemble de scripts utiliser pour réaliser les affichages du rapport. Ces derniers
    doivent être placés à la racine du projet pour pouvoir être exécutés.

### modules du projet ###
    les fichiers auto_encoder.py, linear_module.py, multi_class.py, neural_network.py, non_linear_module.py et sequence.py
    contiennent les classes que nous avons réalisées pour le projet

### tests ###
les fichiers préfixés par "test_" à la racine du projet permettent de tester le bon fonctionnement des classes
précédemment réalisées. Dans la majorité des cas, ils permettent d'afficher l'évolution de l'accuracy et de l'erreur
au fil de l'entraînement des réseaux.