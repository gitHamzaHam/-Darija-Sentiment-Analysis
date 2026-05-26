# 🕌 Darija Sentiment Analysis - Deep Learning

Ce projet est une solution complète d'Analyse de Sentiments (Positif / Négatif / Neutre) pour les avis de produits écrits en **Darija marocain** (en caractères arabes et en Arabizi/alphabet latin), développée pour le module **Apprentissage Profond (Deep Learning)**.

---

## 📊 Le Modèle de Deep Learning

*   **Modèle de base** : `SI2M-Lab/DarijaBERT` (modèle BERT pré-entraîné sur un large corpus de textes marocains).
*   **Version fine-tunée** : `HamzaElhamidineOffi/darija-sentiment-bert` (hébergé sur le Hugging Face Hub).
*   **Performances** : Précision (Accuracy) supérieure à **95%** sur le jeu de test.
*   **Score F1-Macro** : Supérieur à **0.9500** sur le jeu de test équilibré.

---

## 💾 Le Dataset Généré (10 000 avis)

Pour surmonter les biais des datasets publics (souvent très petits et déséquilibrés), le projet utilise un **générateur combinatoire de données** (`generate_dataset.py`) qui produit un jeu de données de **10 000 lignes** :
*   **Équilibre des écritures** : 50% en caractères arabes (عربي), 50% en Arabizi (alphabet latin).
*   **Équilibre des sentiments** : ~3 333 Positifs, ~3 333 Négatifs, ~3 334 Neutres.
*   **Diversité des domaines** : Produits (téléphones, PC, vêtements, etc.), services (livraison, service client) et restauration.

### Variabilité Orthographique du Darija
Le Darija n'ayant pas d'orthographe fixe, le générateur injecte aléatoirement des variantes courantes pour chaque mot-clé :
*   *Beaucoup* : `bzf`, `bzaf`, `bezzaf`, `bzaaf`, `bzaff`.
*   *Bien/Bon* : `mzyan`, `mezyan`, `mziyan`, `mezian`, `mzyane`, `mzyana`.
*   *Mauvais* : `khayb`, `khayba`, `khaib`, `khaser`, `na9s`.

### Mots-Pivots ("walakin" / ولكن)
Pour apprendre au modèle à comprendre les nuances, des phrases neutres ont été conçues en reliant des sentiments opposés via le mot-pivot **"walakin"** (ex: *"had pc zwine walakin ghali chwiya"* $\rightarrow$ Neutre).

### Adjectifs Cumulés
Des structures de phrases renforcent le sentiment en cumulant des adjectifs de même polarité reliés par "et" (ex: *"zwine o wa3er"* $\rightarrow$ Positif, *"khayb o ghali"* $\rightarrow$ Négatif), évitant au modèle d'annuler les sentiments en présence de plusieurs mots porteurs.

---

## 🚀 Utilisation

### 1. Tester l'application Streamlit (Localement)
Pour lancer l'interface web interactive locale et tester le modèle :
1.  Installez les dépendances nécessaires :
    ```bash
    pip install -r requirements.txt
    ```
2.  Démarrez le serveur Streamlit :
    ```bash
    streamlit run app.py
    ```
    *Note : Lors de la première prédiction, le modèle sera téléchargé depuis Hugging Face et mis en cache automatiquement.*

### 2. Ré-entraîner le modèle (Google Colab)
Si vous souhaitez ré-entraîner le modèle sur le jeu de données de 10 000 lignes :
1.  Ouvrez un notebook sur **Google Colab** et activez le **GPU T4** gratuit.
2.  Installez les bibliothèques requises :
    ```python
    !pip install transformers torch pandas scikit-learn huggingface_hub datasets accelerate
    ```
3.  Générez le dataset directement sur Colab (ou importez `darija_dataset_10k.csv`) :
    *   Copiez le code de `generate_dataset.py` et lancez-le :
        ```bash
        !python generate_dataset.py
        ```
4.  Lancez le fine-tuning et l'upload automatique vers Hugging Face :
    *   Importez le script `train_fine_tune.py` et exécutez la configuration suivante :
        ```python
        import sys
        sys.argv = [
            '',
            '--local_csv', 'darija_dataset_10k.csv',
            '--epochs', '3',
            '--batch_size', '16',
            '--hf_token', 'VOTRE_TOKEN_HF_WRITE',
            '--hf_repo', 'VOTRE_NOM_D_UTILISATEUR/darija-sentiment-bert'
        ]
        from train_fine_tune import main
        main()
        ```

---

## 👨‍🎓 Auteurs
*   **Abdellah EL HOUSNI**
*   **HAMZA EL HAMIDINE**
*   **Chaïmae Rady**
*   *Année académique : 2025-2026*
