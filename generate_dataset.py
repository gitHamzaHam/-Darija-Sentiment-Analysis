import random
import csv
import re
import os
import sys

# Configuration de la sortie standard en UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration de la génération
NUM_SAMPLES = 10000
OUTPUT_FILE = "darija_dataset_10k.csv"

# --- LEXIQUE ARABIZI (Alphabet Latin) ---
pos_products_arabizi = ['lproduit', 'had lproduit', 'had sbat', 'tilifoun', 'had tilifoun', 'pc', 'had pc', 'livraison', 'toussil', 'had l\'app', 'lkhidma', 'service', 'hadchi', 'hwayj', 'lhwayj', 'had tomobil', 'had l\'article', 'casque', 'had lmakla', 'had lmahal', 'had ssiara', 'ecouteurs', 'pantalon', 't-shirt', 'srioual', 'dacia', 'tipo']
# Ajout de '3jebni', '3jbatni', '3jbni'
pos_adjectives_arabizi = ['zwine', 'zwin', 'zwina', 'zween', 'top', 'wa3er', 'waer', 'ghzal', 'n9i', 'bnin', 'khedam mzyan', 'khedam mzn', 'satisfait', 'magnifique', 'excellent', 'parfait', 'mlih', 'mzyan', 'mezyan', 'mezian', 'top dial top', 'sria3', 'khfif', 'original', 'mzyana', 'zwine bzaf', 'makaynach bhalha', 'mousali', '3jebni', '3jbatni', '3jbni']
neg_adjectives_arabizi = ['khayb', 'khayba', 'khaib', 'khaser', 'khasra', '3ayan', '3ayana', 'na9s', 'na9sa', 'fchel', 'makhedamch', 'makhedamach', 'zbel', 'ma3jbnich', 'mazyanch', 'mazwinch', 'nul', 'horrible', 'degoutant', 'decu', 'ghali bzaf', 'ghali', 'ghalia', 'ghalia bzaf', 't9il', 't9ila', 'kaytbloqua', 'kaytskhn', 'moussekh', 'mousskha', 'd3if', 'd3ifa', 'mabghitch', 'khasr', 'makanich', 'modir']
neu_adjectives_arabizi = ['3adi', '3adia', 'normal', 'acceptable', 'correct', 'bof', 'moyen', 'moyenne', 'machi chi haja', 'machi khatir', 'sans plus', 'pas exceptionnel', 'kay9di lghard', 'machi khayb', 'machi zwine', 'ordinaire', 'fi lmotawassit', 'mtwsat', '3la 9d lhal', 'na9s chwiya walakin khedam']

intensity_arabizi = ['', 'bzf', 'bzaf', 'bezzaf', 'bzaaf', 'bzaff', 'chwya', 'chwiya', 'chouiya', 'ghi chwiya', 'ga3', 'ga3ma', 'dima', 'vraiment', 'tellement', 'chwiya bzaf']

pos_recs_arabizi = ['nssa7 bih', 'ghir chriwh', 'merci bzaf', 'top bzaf', 'lah i3tikom saha', 'rabi ikhalikom', 'chira2 mzyan', 'je recommande', 'recommande 100%', 'khoudouh wntouma mhaniyin', '3jebni bzaf']
neg_recs_arabizi = ['ndemte 3lih', 'perte d\'argent', 'matkhasrouch flousscom', 'madiwch 3lih', 'jamais de la vie', 'never buy this', '3iyane bzaf', 'ma3awdch', 'khasara dyal lflouss', 'da3a dial lweqt']
neu_recs_arabizi = ['pour le prix ca va', 'pour le prix m9boul', '3adi machi chi haja', 'fait le travail', 'fait le job', 'machi chi haja kbira']

# --- LEXIQUE ARABE (Caractères Arabes) ---
pos_products_arabe = ['المنتج', 'هاد المنتج', 'السروال', 'الصباط', 'التليفون', 'هاد التليفون', 'البيسي', 'التوصيل', 'الخدمة', 'هاد الماكلة', 'هاد التطبيق', 'الحوايج', 'السيارة', 'الطوموبيل', 'المحل', 'هاد المحل', 'السلعة', 'هاد السلعة', 'الكاسك', 'الماكينة']
# Ajout de 'عجبني', 'عجباتني'
pos_adjectives_arabe = ['زوين', 'زوينة', 'مزيان', 'مزيانة', 'طوب', 'واعر', 'واعرة', 'غزال', 'غزالة', 'نقي', 'بنين', 'بنينة', 'خدام مزيان', 'سريع', 'خفيف', 'ممتاز', 'رائع', 'بطل', 'كايعجبني', 'ناجح', 'كناصح بيه', 'ما كاينش بحالو', 'غزال بزاف', 'عجبني', 'عجباتني']
neg_adjectives_arabe = ['خايب', 'خايبة', 'خاسر', 'خاسرة', 'عيان', 'عيانة', 'ناقص', 'ناقصة', 'فشل', 'ماخدامش', 'ماخداماش', 'زبل', 'ما عجبنيش', 'ما مزيانش', 'ما زوينش', 'حامض', 'حامضة', 'غالي بزاف', 'غالي', 'غالية', 'ثقيل', 'ثقيلة', 'كايتبلوكا', 'ضعيف', 'ضعيفة', 'خسارة الفلوس', 'ندمت عليه', 'موسخ', 'موسخة']
neu_adjectives_arabe = ['عادي', 'عادية', 'نورمال', 'مقبول', 'مقبولة', 'متوسط', 'متوسطة', 'ماشي شي حاجة', 'ماشي واعر', 'كيقضي الغرض', 'ماشي khayb', 'ماشي زوين', 'عادي وصافي', 'على قد الحال', 'ناقص شوية ولكن خدام']

intensity_arabe = ['', 'بزاف', 'بزآف', 'بزااف', 'شوية', 'شويية', 'غير شوية', 'كاع', 'كاعma', 'ديما', 'بزاف بزاف', 'فعلا']

pos_recs_arabe = ['كنصح بيه', 'غير شريوه', 'شكرا بزاف', 'الله يعطيك الصحة', 'رائع جدا', 'شراء مميز', 'تستاهل كل خير', 'خودوه وانتوما هانيين', 'عجبني بزاف']
neg_recs_arabe = ['ندمت عليه', 'خسارة الفلوس', 'ما تشريوهش', 'ماتضيعوش فلوسكم فيه', 'ما كايستاهلش', 'توبة نعاود نشري', 'ضياع الوقت والفلوس']
neu_recs_arabe = ['على قد الثمن', 'كيقضي الغرض وصافي', 'عادي وصافي', 'ما بيهش', 'ماشي شي حاجة كبيرة', 'على قد الجيب']

# --- GÉNÉRATEUR ---

def clean_spaces(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_positive_arabizi():
    templates = [
        # [Product] + [Positive Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)} {random.choice(intensity_arabizi)}",
        # [Product] + [Positive Adjective] + [Positive Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)} {random.choice(pos_recs_arabizi)}",
        # [Product] + [Positive Adjective] + [Intensity] + [Positive Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)} {random.choice(intensity_arabizi)} {random.choice(pos_recs_arabizi)}",
        # [Positive Rec] + [Product] + [Positive Adjective]
        lambda: f"{random.choice(pos_recs_arabizi)} {random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)}",
        # [Product] + machi + [Negative Adjective]
        lambda: f"{random.choice(pos_products_arabizi)} machi {random.choice(neg_adjectives_arabizi)} {random.choice(pos_recs_arabizi)}",
        # NOUVEAU: Adjectifs cumulés positifs : [Product] + [Positive Adjective] + o + [Positive Adjective/Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)} o {random.choice(pos_adjectives_arabizi)}"
    ]
    return clean_spaces(random.choice(templates)())

def generate_negative_arabizi():
    templates = [
        # [Product] + [Negative Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)} {random.choice(intensity_arabizi)}",
        # [Product] + [Negative Adjective] + [Negative Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)} {random.choice(neg_recs_arabizi)}",
        # [Product] + [Negative Adjective] + [Intensity] + [Negative Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)} {random.choice(intensity_arabizi)} {random.choice(neg_recs_arabizi)}",
        # [Negative Rec] + [Product] + [Negative Adjective]
        lambda: f"{random.choice(neg_recs_arabizi)} {random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)}",
        # [Product] + machi + [Positive Adjective]
        lambda: f"{random.choice(pos_products_arabizi)} machi {random.choice(pos_adjectives_arabizi)} {random.choice(neg_recs_arabizi)}",
        # NOUVEAU: Adjectifs cumulés négatifs : [Product] + [Negative Adjective] + o + [Negative Adjective]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)} o {random.choice(neg_adjectives_arabizi)}"
    ]
    return clean_spaces(random.choice(templates)())

def generate_neutral_arabizi():
    templates = [
        # [Product] + [Neutral Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neu_adjectives_arabizi)} {random.choice(intensity_arabizi)}",
        # [Product] + [Neutral Adjective] + [Neutral Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neu_adjectives_arabizi)} {random.choice(neu_recs_arabizi)}",
        # Pivot "walakin" : [Product] + [Positive Adjective] + walakin + [Negative Adjective/Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(pos_adjectives_arabizi)} walakin {random.choice(neg_adjectives_arabizi)}",
        # Pivot "walakin" : [Product] + [Negative Adjective] + walakin + [Positive Adjective/Rec]
        lambda: f"{random.choice(pos_products_arabizi)} {random.choice(neg_adjectives_arabizi)} walakin {random.choice(pos_adjectives_arabizi)}",
        # [Product] + machi + [Positive Adjective] + walakin + [Neutral Rec]
        lambda: f"{random.choice(pos_products_arabizi)} machi {random.choice(pos_adjectives_arabizi)} walakin {random.choice(neu_recs_arabizi)}"
    ]
    return clean_spaces(random.choice(templates)())

def generate_positive_arabe():
    templates = [
        # [Product] + [Positive Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)} {random.choice(intensity_arabe)}",
        # [Product] + [Positive Adjective] + [Positive Rec]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)} {random.choice(pos_recs_arabe)}",
        # [Product] + [Positive Adjective] + [Intensity] + [Positive Rec]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)} {random.choice(intensity_arabe)} {random.choice(pos_recs_arabe)}",
        # [Positive Rec] + [Product] + [Positive Adjective]
        lambda: f"{random.choice(pos_recs_arabe)} {random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)}",
        # [Product] + ماشي + [Negative Adjective] + [Positive Rec]
        lambda: f"{random.choice(pos_products_arabe)} ماشي {random.choice(neg_adjectives_arabe)} {random.choice(pos_recs_arabe)}",
        # NOUVEAU: Adjectifs cumulés positifs : [Product] + [Positive Adjective] + و + [Positive Adjective]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)} و {random.choice(pos_adjectives_arabe)}"
    ]
    return clean_spaces(random.choice(templates)())

def generate_negative_arabe():
    templates = [
        # [Product] + [Negative Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)} {random.choice(intensity_arabe)}",
        # [Product] + [Negative Adjective] + [Negative Rec]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)} {random.choice(neg_recs_arabe)}",
        # [Product] + [Negative Adjective] + [Intensity] + [Negative Rec]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)} {random.choice(intensity_arabe)} {random.choice(neg_recs_arabe)}",
        # [Negative Rec] + [Product] + [Negative Adjective]
        lambda: f"{random.choice(neg_recs_arabe)} {random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)}",
        # [Product] + ماشي + [Positive Adjective] + [Negative Rec]
        lambda: f"{random.choice(pos_products_arabe)} ماشي {random.choice(pos_adjectives_arabe)} {random.choice(neg_recs_arabe)}",
        # NOUVEAU: Adjectifs cumulés négatifs : [Product] + [Negative Adjective] + و + [Negative Adjective]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)} و {random.choice(neg_adjectives_arabe)}"
    ]
    return clean_spaces(random.choice(templates)())

def generate_neutral_arabe():
    templates = [
        # [Product] + [Neutral Adjective] + [Intensity]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neu_adjectives_arabe)} {random.choice(intensity_arabe)}",
        # [Product] + [Neutral Adjective] + [Neutral Rec]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neu_adjectives_arabe)} {random.choice(neu_recs_arabe)}",
        # Pivot "walakin" : [Product] + [Positive Adjective] + ولكن + [Negative Adjective]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(pos_adjectives_arabe)} ولكن {random.choice(neg_adjectives_arabe)}",
        # Pivot "walakin" : [Product] + [Negative Adjective] + ولكن + [Positive Adjective]
        lambda: f"{random.choice(pos_products_arabe)} {random.choice(neg_adjectives_arabe)} ولكن {random.choice(pos_adjectives_arabe)}",
        # [Product] + ماشي + [Positive Adjective] + ولكن + [Neutral Rec]
        lambda: f"{random.choice(pos_products_arabe)} ماشي {random.choice(pos_adjectives_arabe)} ولكن {random.choice(neu_recs_arabe)}"
    ]
    return clean_spaces(random.choice(templates)())

def main():
    print(f"Génération de {NUM_SAMPLES} avis Darija...")
    unique_reviews = set()
    rows = []
    target_per_sentiment = NUM_SAMPLES // 3
    counts = {"positif": 0, "negatif": 0, "neutre": 0}
    scripts = ["arabe", "arabizi"]
    attempts = 0
    max_attempts = NUM_SAMPLES * 50
    
    while len(rows) < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1
        sentiment = random.choice(["positif", "negatif", "neutre"])
        if counts[sentiment] >= target_per_sentiment and len(rows) < (NUM_SAMPLES - 2):
            continue
        script = random.choice(scripts)
        if script == "arabizi":
            if sentiment == "positif":
                review = generate_positive_arabizi()
            elif sentiment == "negatif":
                review = generate_negative_arabizi()
            else:
                review = generate_neutral_arabizi()
        else:
            if sentiment == "positif":
                review = generate_positive_arabe()
            elif sentiment == "negatif":
                review = generate_negative_arabe()
            else:
                review = generate_neutral_arabe()
                
        if review not in unique_reviews and len(review.split()) >= 2:
            unique_reviews.add(review)
            rows.append({"review": review, "sentiment": sentiment})
            counts[sentiment] += 1

    random.shuffle(rows)
    with open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["review", "sentiment"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Génération terminée avec succès !")
    print(f"Fichier sauvegardé sous : {OUTPUT_FILE}")
    print(f"Statistiques :")
    print(f"  - Total lignes : {len(rows)}")
    print(f"  - Positifs : {counts['positif']}")
    print(f"  - Négatifs : {counts['negatif']}")
    print(f"  - Neutres : {counts['neutre']}")

if __name__ == "__main__":
    main()
