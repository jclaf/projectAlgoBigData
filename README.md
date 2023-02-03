# projectAlgoBigData

Ce projet présente un ensemble de modèle d'apprentissage afin de classifier les données lié à la pollution mondiale obtenu sur Kaggle 
https://www.kaggle.com/datasets/zvr842/global-pollution-by-counties

## Modèle d'apprentissage

J'utilise les 4 modèles supervisé :
    
    - Arbre de décision
    - Forêt aléatoire
    - Regression Linéaire
    - Gradient-boost tree regression

et 2 modèles non supervisé :
    
    - Kmeans
    - Gaussian Mixture Means
## Structure du projet

Dans le dossier "data", on y trouve :
    
    - le dossier "image" 
    - le dataset
    - les données annexes pouvant y être sauvegarder

Le dossier "src" contient un ensemble de dossier lié au projet et le fichier main.py :
    
    - ACP : contient les fonctions du PCA
    - AlgorithmModelling : contient 3 fichiers (classification, regression, clustering )
    - DataExploration : contient les fonctions permettant de nettoyer les données et de créer le vecteur "features"
    - Utils: les fonctions réutilisable

Le fichier "Classification.py" contient les algorithmes de l'arbre de décision et la forêt aléatoire.

Le fichier "Regression.py" contient les algorithmes de la régression linéaire et le gradient-boost tree regression.

Le fichier "Clustering.py" contient les algorithmes de "Kmeans" et "Gaussian Mixture Means"

Le dossier "Notebook" contient un fichier .ipynb du projet.

## Execution

Pour faire tourner le programme, il suffit de faire python main.py ou python3 main.py

Il est également possible de faire appel à la dataset depuis les arguments .

On peut éxécuter le fichier exec.sh pour faire tourner le programme. 
