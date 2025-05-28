# 🖐 Détection de Mains avec OpenCV, MediaPipe et CvZone

## 🎯 Objectif

Ce projet te permet de **détecter tes mains en temps réel** avec la webcam, de suivre la position de tes doigts et de mesurer la distance entre eux.

Il est conçu pour apprendre **pas à pas** à utiliser la **vision par ordinateur** en Python.

---

## 🧰 Outils utilisés

- **PyCharm** : un éditeur de code pour Python
- **Python** (version 3.8+)
- **OpenCV** : pour gérer les images et la webcam
- **MediaPipe** : pour détecter les mains
- **CvZone** : pour dessiner facilement sur l’image
- **Environnement virtuel (`.venv`)** déjà configuré dans le projet

---

## 🧑‍🏫 Étapes pour ouvrir et lancer le projet dans PyCharm

1. **Ouvre PyCharm**  
   Clique sur `Open` et sélectionne le dossier du projet.

2. **Active l’environnement virtuel `.venv`** :
   - En bas à droite, clique sur le message `Python interpreter`.
   - Clique sur `Add Interpreter > Add Local Environment`.
   - Clique sur `select existing` apres sur  `OK` .
   - Clique sur OK.


Voici un petit exercice pour t'entraîner à modifier le code :

🧠 Objectif : Afficher un "X" ou un "O" dans le rectangle vide selon la distance entre l’index et le majeur.

📝 Consigne :

Si la distance length est inférieure à 20, afficher un "X" dans le rectangle.

Sinon, afficher un "O".
