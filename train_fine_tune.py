import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import argparse
import os
import sys

# Prétraitement Darija
class DarijaPreprocessor:
    def __init__(self):
        # Pour les modèles de Deep Learning (BERT), on ne supprime pas de mots
        # pour conserver tout le contexte (négations comme machi/ma/la, pivots comme walakin, etc.)
        self.stopwords = set()
    
    def normalize(self, text):
        text = str(text).lower()
        # Normalisation Arabizi
        text = re.sub(r'3', 'a', text)
        text = re.sub(r'7', 'h', text)
        text = re.sub(r'9', 'q', text)
        text = re.sub(r'2', 'a', text)
        # Suppression répétitions
        text = re.sub(r'(.)\1+', r'\1', text)
        # Suppression ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Suppression espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)
    
    def preprocess(self, text):
        text = self.normalize(text)
        text = self.remove_stopwords(text)
        return text

# Dataset PyTorch
class DarijaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

def main():
    # Configuration de la sortie standard en UTF-8 pour éviter les erreurs d'encodage sur Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Fine-tuning de DarijaBERT / XLM-R")
    parser.add_argument("--model_name", type=str, default="SI2M-Lab/DarijaBERT", help="Nom du modèle pré-entraîné")
    parser.add_argument("--hf_token", type=str, default="", help="Token d'écriture Hugging Face")
    parser.add_argument("--hf_repo", type=str, default="", help="Nom du repo de destination sur HF (ex: 'pseudo/darija-sentiment')")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size d'entraînement")
    parser.add_argument("--dataset_name", type=str, default="", help="Nom du dataset sur Hugging Face (ex: 'ohidaoui/darija-reviews')")
    parser.add_argument("--local_csv", type=str, default="darija_dataset_10k.csv", help="Chemin vers un fichier CSV local")
    parser.add_argument("--text_column", type=str, default="review", help="Nom de la colonne contenant le texte")
    parser.add_argument("--label_column", type=str, default="sentiment", help="Nom de la colonne contenant les sentiments/labels")
    
    args, _ = parser.parse_known_args()
    
    print("🚀 Chargement des données...")
    if args.dataset_name:
        from datasets import load_dataset
        print(f"🤗 Chargement du dataset Hugging Face : {args.dataset_name}")
        raw_ds = load_dataset(args.dataset_name)
        split_name = list(raw_ds.keys())[0]
        df = pd.DataFrame(raw_ds[split_name])
    elif os.path.exists(args.local_csv):
        print(f"📂 Chargement du fichier CSV local : {args.local_csv}")
        df = pd.read_csv(args.local_csv)
    else:
        raise FileNotFoundError(f"Fichier CSV spécifié introuvable : {args.local_csv}")

    # Normalisation dynamique des noms de colonnes
    if args.text_column != "review" and args.text_column in df.columns:
        df = df.rename(columns={args.text_column: "review"})
    if args.label_column != "sentiment" and args.label_column in df.columns:
        df = df.rename(columns={args.label_column: "sentiment"})

    preprocessor = DarijaPreprocessor()
    print("🧹 Nettoyage et prétraitement du texte...")
    df['clean_review'] = df['review'].apply(preprocessor.preprocess)

    # Normalisation robuste des étiquettes (positif/négatif/neutre ou positive/negative/neutral ou 2/0/1)
    def normalize_label(val):
        val_str = str(val).lower().strip()
        if "posit" in val_str or val_str == "2":
            return 2
        elif "negat" in val_str or val_str == "0":
            return 0
        else:
            return 1

    df['label'] = df['sentiment'].apply(normalize_label)

    # Équilibrage des classes (Oversampling)
    print("⚖️ Équilibrage des classes (Oversampling)...")
    max_size = df['label'].value_counts().max()
    balanced_dfs = []
    for class_index, group in df.groupby('label'):
        balanced_dfs.append(group.sample(max_size, replace=True))
    df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df['clean_review'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"📊 Dataset final équilibré : {len(df)} exemples. Train: {len(X_train)}, Test: {len(X_test)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Entraînement sur : {device}")
    
    print(f"📦 Téléchargement du tokenizer et du modèle : {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    train_dataset = DarijaDataset(X_train.tolist(), y_train, tokenizer)
    test_dataset = DarijaDataset(X_test.tolist(), y_test, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("🔥 Lancement du Fine-Tuning...")
    trainer.train()
    
    print("📊 Évaluation finale du modèle...")
    eval_results = trainer.evaluate()
    print(f"✅ Accuracy finale : {eval_results['eval_accuracy']:.4f}")
    print(f"✅ F1-Macro finale : {eval_results['eval_f1_macro']:.4f}")
    
    # Sauvegarde locale
    os.makedirs("./models", exist_ok=True)
    model.save_pretrained("./models/darija_fine_tuned")
    tokenizer.save_pretrained("./models/darija_fine_tuned")
    print("💾 Modèle sauvegardé localement sous './models/darija_fine_tuned'")
    
    # Upload sur Hugging Face si token spécifié
    if args.hf_token and args.hf_repo:
        from huggingface_hub import login
        print("🤗 Connexion à Hugging Face...")
        login(token=args.hf_token)
        print(f"📤 Upload du modèle vers {args.hf_repo}...")
        model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        print("✅ Modèle publié sur Hugging Face avec succès !")

if __name__ == "__main__":
    main()
