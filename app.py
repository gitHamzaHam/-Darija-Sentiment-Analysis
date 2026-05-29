import streamlit as st
from transformers import pipeline
import re

# Définition du préprocesseur (conservé pour afficher le texte nettoyé à des fins de comparaison)
class DarijaPreprocessor:
    """Prétraitement personnalisé pour le Darija"""
    
    def __init__(self):
        # Pour les modèles de Deep Learning (BERT), on ne supprime pas de mots
        # pour conserver tout le contexte (négations comme machi/ma/la, pivots comme walakin, etc.)
        self.stopwords = set()
    
    def normalize(self, text):
        """Normalisation du texte"""
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
        """Suppression des stopwords"""
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)
    
    def preprocess(self, text):
        """Pipeline complet"""
        text = self.normalize(text)
        text = self.remove_stopwords(text)
        return text

@st.cache_resource
def load_classifier(model_name):
    # Charge le pipeline d'analyse de sentiments Hugging Face
    return pipeline("sentiment-analysis", model=model_name)

def normalize_label(pred_label, model_name):
    label = str(pred_label).lower()
    
    # Cas spécifique pour le modèle par défaut monsifnadir (LABEL_0=Neutre, LABEL_1=Négatif, LABEL_2=Positif)
    if "monsifnadir" in model_name.lower():
        if "2" in label:
            return "positif"
        elif "1" in label:
            return "negatif"
        else:
            return "neutre"
            
    # Cas général basé sur les mots clés (posit, negat, neut)
    if "posit" in label:
        return "positif"
    elif "negat" in label:
        return "negatif"
    elif "neut" in label:
        return "neutre"
        
    # Fallback si ce sont des étiquettes brutes LABEL_0, LABEL_1, LABEL_2
    if "0" in label:
        return "negatif"
    elif "1" in label:
        return "neutre"
    else:
        return "positif"


def main():
    st.set_page_config(page_title="Darija Sentiment Analysis", page_icon="🇲🇦", layout="centered")

    # Style CSS personnalisé haut de gamme (Accents marocains modernes : vert émeraude doux, rouge bordeaux discret, cartes douces)
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
            
            /* Appliquer la police Outfit */
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            
            /* Titre principal */
            .main-title {
                font-size: 2.8rem;
                font-weight: 800;
                background: linear-gradient(135deg, #006241 0%, #c1272d 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                font-size: 1.1rem;
                color: #555555;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            /* Style des cartes de résultats */
            .result-card {
                padding: 1.5rem;
                border-radius: 16px;
                border-left: 6px solid;
                margin-top: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }
            .result-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.1);
            }
            
            .pos-card {
                background-color: #eefdf4;
                border-left-color: #10b981;
                color: #065f46;
            }
            .neg-card {
                background-color: #fdf2f2;
                border-left-color: #ef4444;
                color: #991b1b;
            }
            .neu-card {
                background-color: #fefaf0;
                border-left-color: #f59e0b;
                color: #92400e;
            }
            
            /* Badges */
            .custom-badge {
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.85rem;
                display: inline-block;
                margin-top: 0.5rem;
            }
            .badge-pos { background-color: #10b981; color: white; }
            .badge-neg { background-color: #ef4444; color: white; }
            .badge-neu { background-color: #f59e0b; color: white; }
            
            /* Boutons d'exemples */
            .stButton>button {
                border-radius: 8px;
                border: 1px solid #d1d5db;
                background-color: white;
                color: #374151;
                font-size: 0.85rem;
                padding: 0.4rem 0.8rem;
                transition: all 0.2s;
                width: 100%;
                text-align: left;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .stButton>button:hover {
                border-color: #006241;
                color: #006241;
                background-color: #f0fdf4;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🇲🇦 Darija Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyseur de sentiments basé sur un modèle Deep Learning fine-tuné sur 10 000 avis Jumia/Facebook en Darija (arabe et latin).</div>', unsafe_allow_html=True)

    # Paramètres du modèle en barre latérale
    st.sidebar.header("⚙️ Configuration")
    default_model = "HamzaElhamidineOffi/darija-sentiment-bert"
    model_name = st.sidebar.text_input("Modèle Hugging Face :", value=default_model)
    
    preprocessor = DarijaPreprocessor()

    # Chargement du modèle de Deep Learning
    try:
        with st.spinner("Chargement du modèle de Deep Learning... (veuillez patienter la première fois)"):
            classifier = load_classifier(model_name)
        st.sidebar.success(f"✅ Modèle '{model_name.split('/')[-1]}' chargé !")
    except Exception as e:
        st.sidebar.error(f"Erreur de chargement : {e}")
        st.stop()

    # Gestion de l'état pour les exemples en session_state
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    st.subheader("💡 Exemples rapides à tester")
    st.caption("Cliquez sur un bouton pour charger l'exemple correspondant :")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("**🟢 Avis Positifs**")
        if st.button("ariel top (Arabizi)", key="btn_pos_1"):
            st.session_state.text_input = "ariel zwin bzaf kaykhli riha top"
        if st.button("تليفون واعر (Arabe)", key="btn_pos_2"):
            st.session_state.text_input = "هاد التليفون واعر بزاف وخدام مزيان"
            
    with cols[1]:
        st.markdown("**🔴 Avis Négatifs**")
        if st.button("omo khayb (Arabizi)", key="btn_neg_1"):
            st.session_state.text_input = "omo khayb makiqych hwayj"
        if st.button("توصيل تعطل (Arabe)", key="btn_neg_2"):
            st.session_state.text_input = "التوصيل تعطل بزاف والسلعة ناقصة"
            
    with cols[2]:
        st.markdown("**🟡 Avis Neutres**")
        if st.button("produit cher (Arabizi)", key="btn_neu_1"):
            st.session_state.text_input = "lproduit zwine walakin ghali chwiya"
        if st.button("تليفون عادي (Arabe)", key="btn_neu_2"):
            st.session_state.text_input = "التليفون عادي كيقضي الغرض وصافي"

    st.markdown("---")
    
    st.subheader("✍️ Entrez votre commentaire")
    user_input = st.text_area("Saisissez votre commentaire en Darija :", value=st.session_state.text_input, height=100)

    # Synchroniser l'état si l'utilisateur saisit manuellement
    if user_input != st.session_state.text_input:
        st.session_state.text_input = user_input

    if st.button("Analyser le sentiment 🚀", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Veuillez entrer un texte ou choisir un exemple avant d'analyser.")
        else:
            with st.spinner("Analyse sémantique en cours..."):
                # Prétraitement
                processed_text = preprocessor.preprocess(user_input)
                
                # Prédiction
                results = classifier(processed_text)
                prediction = results[0]['label']
                score = results[0]['score']
                
                # Normalisation du label
                pred_str = normalize_label(prediction, model_name)
                
                # Rendu visuel haut de gamme
                if pred_str == 'positif':
                    st.markdown(f"""
                        <div class="result-card pos-card">
                            <h3>🟢 Sentiment Positif</h3>
                            <p>Le modèle a détecté un avis favorable.</p>
                            <span class="custom-badge badge-pos">Confiance : {score:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                elif pred_str == 'negatif':
                    st.markdown(f"""
                        <div class="result-card neg-card">
                            <h3>🔴 Sentiment Négatif</h3>
                            <p>Le modèle a détecté une opinion critique ou insatisfaite.</p>
                            <span class="custom-badge badge-neg">Confiance : {score:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card neu-card">
                            <h3>🟡 Sentiment Neutre / Mitigé</h3>
                            <p>Le modèle a détecté un sentiment nuancé.</p>
                            <span class="custom-badge badge-neu">Confiance : {score:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Informations techniques complémentaires
                with st.expander("🔍 Voir les détails techniques du prétraitement"):
                    st.markdown(f"**Texte brut saisi :** `{user_input}`")
                    st.markdown(f"**Texte normalisé & nettoyé :** `{processed_text}`")
                    st.markdown(f"**Label brut du modèle :** `{prediction}`")

if __name__ == "__main__":
    main()
