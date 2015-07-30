# Identification d'images pour la botanique

## Présentation

Projet initié par Mohamed Issolah, repris par Vincent Montalieu dans le cadre d'un stage de 4ème année d'école d'ingénieur.

*Polytech Nice-Sophia*

*I3S - Laboratoire d'Informatique, Signaux et Systèmes de Sophia Antipolis*

## Architecture

* Le dossier contenant les données doit recevoir tous les fichiers JPG et XML puis être mis en place en
utilisant le script `./scaffolding.sh` présent dans le dossier *scripts*

* Le nom du dossier contenant les données est libre.

## Code

### vocabulary.cpp

* Crée un vocabulaire général commun à la base de training avec k mots provenants du k-means.

* Utilisation : `./vocabulary FeatureDetector DescriptorExtractor DescriptorMatcher Clusters DataFolder`