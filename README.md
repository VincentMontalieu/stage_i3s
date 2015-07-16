# stage_i3s

------------------------------------------------------------------------------------------------------
*** Architecture ***

Le dossier contenant les données doit recevoir tous les fichiers JPG et XML puis être mis en place en
utilisant le script training_test_separator_data_generator.sh

Le nom du dossier contenant les données est libre
------------------------------------------------------------------------------------------------------

*** vocabulary.cpp ***

- Crée un vocabulaire général commun à la base de training avec k mots provenants du k-means
- EXEC : ./vocabulary SIFT SIFT BruteForce 100 ../../my_data/