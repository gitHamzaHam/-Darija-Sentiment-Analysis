# 🕌 Darija Sentiment Analysis - Deep Learning

Ce projet est une solution de pointe pour l'**Analyse de Sentiments (Positif / Négatif / Neutre)** dans les avis écrits en **Darija marocain** (en caractères arabes et en Arabizi/alphabet latin). Développé dans le cadre du module **Apprentissage Profond (Deep Learning)**, ce système exploite la puissance des architectures Transformers grâce à un modèle **DarijaBERT** fine-tuné sur un dataset sur-mesure de **10 000 exemples**.

---

## 📊 Le Modèle de Deep Learning

*   **Architecture de base** : `SI2M-Lab/DarijaBERT` (modèle BERT pré-entraîné sur un large corpus de textes marocains).
*   **Version fine-tunée** : `HamzaElhamidineOffi/darija-sentiment-bert` (hébergé sur le Hugging Face Hub).
*   **Performances** : Précision (Accuracy) supérieure à **95%** sur le jeu de test équilibré.

---

## 💾 Le Dataset Généré (10 000 avis)

Pour surmonter les biais des datasets publics (souvent très petits et déséquilibrés), le projet utilise un **générateur combinatoire de données** (`generate_dataset.py`) qui produit un jeu de données hautement représentatif de **10 000 lignes** :
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

## 🧹 Le Prétraitement et son Correctif Critique

Les données passent par un prétraitement (`DarijaPreprocessor`) avant d'être envoyées au modèle :
1.  **Normalisation** : Conversion en minuscules, standardisation des chiffres Arabizi (`3` $\rightarrow$ `a`, `7` $\rightarrow$ `h`, `9` $\rightarrow$ `q`, `2` $\rightarrow$ `a`).
2.  **Suppression des répétitions** : Réduction des lettres doublées dues à l'intonation (ex: `bzaaaaf` $\rightarrow$ `bzaf`).
3.  **Préservation des Stopwords (Décision Critique)** : 
    > [!IMPORTANT]
    > Pour ce modèle BERT, la suppression des mots vides (stopwords) a été **désactivée**. La suppression de mots comme `'machi'` (négative : "pas"), `'ma'`/`'la'` (négatives : "non") ou `'walakin'` (pivot : "mais") détruisait le contexte des phrases en Arabizi à l'entraînement. Les préserver garantit une précision maximale.

---

## 🎨 L'Interface Streamlit (`app.py`)

L'application intègre un design premium inspiré de la charte moderne marocaine (vert émeraude doux, rouge bordeaux discret, typographie Outfit de Google Fonts) :
*   **Boutons rapides d'exemples** : Cliquez pour charger instantanément des phrases types (positives, négatives et neutres en arabe et en latin).
*   **Analyse sémantique en temps réel** : Affiche le sentiment prédit, le score de confiance et des explications visuelles.
*   **Volet dépliable technique** : Permet de voir le texte nettoyé après prétraitement et le label brut renvoyé par le réseau de neurones.

---

## 📁 Structure des fichiers

```text
-Darija-Sentiment-Analysis/
├── app.py                      # Application web Streamlit (Deep Learning)
├── generate_dataset.py         # Script générateur du dataset synthétique (10k lignes)
├── train_fine_tune.py          # Script autonome de fine-tuning & publication Hugging Face
├── darija_dataset_10k.csv      # Le jeu de données généré (10 000 exemples)
├── requirements.txt            # Dépendances requises du projet
└── README.md                   # Ce document explicatif
```

---

## 🚀 Lancement Rapide

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Démarrer l'application locale
```bash
python -m streamlit run app.py
```

### 3. Ré-entraîner le modèle sur Google Colab
Consultez le fichier [walkthrough.md](.system_generated/logs/walkthrough.md) ou votre dossier d'artefacts pour obtenir les cellules prêtes à l'emploi (avec GPU T4 gratuit) pour ré-entraîner et publier le modèle en 1 clic.

---

## 👨‍🎓 Auteurs
*   **Abdellah EL HOUSNI**
*   **HAMZA EL HAMIDINE**
*   **Chaïmae Rady**
*   *Année académique : 2025-2026 - Module Deep Learning*
