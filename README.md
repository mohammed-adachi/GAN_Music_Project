# 🎼 APPASSIONATO : Compositeur de Musique par IA
**APPASSIONATO** est un logiciel capable de composer de la musique originale multi-pistes (Piano, Guitare, Basse, Cordes, Batterie).
Le projet gère l'intégralité du flux de travail : de la collecte automatisée de données sur le web jusqu'à la génération de fichiers MIDI via une interface utilisateur interactive.

## 📂 Structure du Projet

Ce dépôt est organisé en modules indépendants, chacun gérant une étape spécifique du processus :

| Fichier | Description |
| :--- | :--- |
| **`scrape_bitmidi.py`** | **Collecte de Données.** Un script automatisé pour télécharger des fichiers MIDI depuis le web (BitMidi) afin de construire le dataset. |
| **`generator.py`** | **L'Artiste.** Contient l'architecture du réseau de neurones responsable de la création pure de la musique. |
| **`discriminator.py`** | **Le Critique.** Contient l'architecture qui évalue la qualité, le rythme et l'harmonie des pistes générées. |
| **`entrainement.ipynb`** | **L'Atelier.** Un notebook Jupyter pour exécuter la boucle d'entraînement où le Générateur et le Discriminateur apprennent ensemble. |
| **`app_streamlit.py`** | **Le Studio.** Une interface web (GUI) pour interagir avec le modèle entraîné, générer de la musique et l'écouter. |

---

## 🛠️ Installation

1. **Cloner le dépôt :**
   git clone https://github.com/mohammed-adachi/GAN_Music_Project.git
   cd GAN_Music_Project
   ```bash
   git clone https://github.com/mohammed-adachi/GAN_Music_Project.git
   cd GAN_Music_Project
    Utilisation du Logiciel
   Étape 1 : Collecte de Données (Optionnel)
   python scrape_bitmidi.py
   Étape 2 : Entraînement
   Pour lancer l'apprentissage de l'IA :
   - Ouvrez entrainement.ipynb dans Jupyter Notebook ou Google Colab.
   - Exécutez toutes les cellules.
   - Une fois terminé, le modèle sauvegardera ses poids dans un fichier nommé musegan_checkpoint.pth.
   Étape 3 : Lancer l'Application (Démo)
   - streamlit run app_streamlit.py
