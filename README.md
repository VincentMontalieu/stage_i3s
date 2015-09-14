# Identification d'images pour la botanique

## Présentation

Projet initié par Mohamed Issolah, repris par Vincent Montalieu dans le cadre d'un stage de 4ème année d'école d'ingénieur.

*Polytech Nice-Sophia*

*I3S - Laboratoire d'Informatique, Signaux et Systèmes de Sophia Antipolis*

## Architecture des données

* Le dossier contenant les données doit recevoir tous les fichiers JPG et XML puis être mis en place en suivant les étapes décrites plus loin.

* Le nom du dossier contenant les données est libre.

### Mise en place des données

1. Se rendre dans le dossier contenant les données en vrac et y exécutant le script `separator.sh`, les données sont alors séparées en un ensemble de train et un ensemble de test.

2. Se rendre dans le dossier *training* puis y exécuter le script `organize_type.sh`, les données de train sont alors séparées par organe.

3.  Se rendre dans chaque dossier organe (branch, entire, flower, fruit, leaf, stem) et y exécuter le script `scaffolding.sh`, l'architecture interne est alors générée et une répartition aléatoire des données entre train (80%) et test (20%) est effectuée. Les fichiers `training.data` et `testing.data` sont générés.

## Code

### Vocabulary.cpp

* Crée un vocabulaire (dictionnaire) général commun à la base d'apprentissage puis génère les histogrammes des images à partir du dictionnaire créé.

* Cette étape est la première de la chaîne de traitement. Elle permet de générer des données normalisées pour chaque image de la base d'apprentissage et ainsi pouvoir effectuer la suite du traitement sur des données stables (des fichiers XML).

* Un log est fournit dans le dossier *results* contenant la durée d'exécution du module.

* Utilisation : `./vocabulary.out Data_Folder Nombre_Clusters`

### Svm.cpp

* Entraîne un SVM pour chaque classe (espèce de plante) de la base.

* Cette étape est la seconde de la chaîne de traitement. Elle permet de générer des SVM qui serviront ensuite de comparateurs lors de la reconnaissance.

* Un log est fournit dans le dossier *results* contenant la durée d'exécution du module.

* Utilisation : `./svm.out Data_Folder Nombre_Clusters C`

### Testing.cpp

* Effectue un test pour toutes les images présentes dans le dossier *testing* et fournit un fichier de log dans le dossier *results* contenant le score global ainsi que la durée du test.

* Utilisation : `./testing.out Data_Folder Nombre_Clusters C`

### TestingSingleFile.cpp

* Effectue un test pour un fichier image spécifique et produit un fichier de log au format JSON. Ce module est utilisé par l'application web.

* Utilisation : `./testing_single_file.out Data_folder Nbr_cluster C Image_to_analyze Log_file`

#### ImageData.cpp

* Représente un objet contenant les points d'intérêts et les descripteurs d'une image.

* Utilisée dans `Vocabulary.cpp` pour éviter d'effectuer deux fois le chargement puis l'analyse des images. Un objet **ImageData** est attribué à chaque image lors de la création du vocabulaire puis réutilisé lors de la création des histogrammes.

#### Soft.cpp

* Représente une implémentation d'un extracteur de BOW en soft assignment, en respectant l'implémentation OpenCV de la version hard assignment.

* Utilisée dans `Vocabulary.cpp` lors de la création des histogrammes (méthode des KNN).

#### Tools.cpp

* Fichier permettant d'éviter la duplication de code. Contient des méthodes générales utilisées par plusieurs modules de la chaîne de traitement.