# 🕌 Darija Sentiment Analysis

## Projet Deep Learning - Analyse de sentiments en Darija Marocain

### 🎯 Objectif
Développer un modèle NLP pour analyser les sentiments (positif/négatif/neutre) des avis produits sur **Jumia.ma** en Darija.

### 📊 Résultats

| Modèle | Accuracy |
|--------|----------|
| Logistic Regression (Classique) | 100% |
| XLM-RoBERTa (Deep Learning) | 32% |

### 📁 Structure du projet
Darija_Sentiment_Project/
├── 00_Projet_Darija_Sentiment_Analysis.ipynb # Notebook complet
├── models/
│ ├── darija_sentiment_model.pkl # Modèle classique
│ ├── vectorizer.pkl # Vectoriseur TF-IDF
│ └── preprocessor.pkl # Prétraitement Darija
├── reports/
│ └── final_results.json # Résultats
└── *.png # Graphiques

### 🚀 Utilisation
# Charger le modèle
import joblib
model = joblib.load('models/darija_sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Prédire un sentiment
text = "had produit zwina bzf"
processed = preprocessor.preprocess(text)
vectorized = vectorizer.transform([processed])
prediction = model.predict(vectorized)[0]
# 0: Négatif, 1: Neutre, 2: Positif
⚠️ Note sur le modèle Deep Learning
Le modèle XLM-RoBERTa (1.06 GB) dépasse la limite de GitHub (100 MB).
Il n'est donc pas inclus dans ce dépôt. Le modèle classique est entièrement disponible.

👨‍🎓 Auteur
Abdellah EL HOUSNI - HAMZA EL HAMIDINE
Année académique: 2025-2026
