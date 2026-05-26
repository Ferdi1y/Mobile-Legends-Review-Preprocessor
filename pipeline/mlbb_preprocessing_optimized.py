#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
MLBB SENTIMENT PREPROCESSING - OPTIMIZED PIPELINE
============================================================
Optimasi:
1. ✅ Kamus slang diperluas 3x lipat (150+ entri)
2. ✅ Fungsi stemming terintegrasi untuk modeling tradisional
3. ✅ Logic labeling yang konsisten
4. ✅ Kolom `review_stemmed` khusus untuk ML models
============================================================
"""

import pandas as pd
import numpy as np
import re
import emoji
import warnings
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

warnings.filterwarnings('ignore')
tqdm.pandas()

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# ============================================================
# KONFIGURASI: KAMUS SLANG DIPERLUAS (150+ ENTRI)
# ============================================================

SLANG_DICT_OPTIMIZED = {
    # === Typo nama game/perusahaan ===
    'monton': 'moonton', 'muntun': 'moonton', 'munton': 'moonton',
    'montoon': 'moonton', 'montun': 'moonton', 'montonn': 'moonton',
    'ml': 'mobile legends', 'mlbb': 'mobile legends',
    'geme': 'game', 'gama': 'game', 'gem': 'game',
    
    # === Singkatan umum ===
    'gw': 'saya', 'gua': 'saya', 'w': 'saya', 'aku': 'saya', 'akuu': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'u': 'kamu', 'km': 'kamu', 'ente': 'kamu',
    'yg': 'yang', 'yng': 'yang', 'yng': 'yang', 'ygn': 'yang',
    'gak': 'tidak', 'ngak': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'ga': 'tidak', 'enggak': 'tidak', 'nope': 'tidak', 'ngg': 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'bget': 'banget', 'bgt': 'banget',
    'dgn': 'dengan', 'dg': 'dengan', 'dgan': 'dengan',
    'aja': 'saja', 'aj': 'saja', 'ajh': 'saja',
    'jg': 'juga', 'jga': 'juga', 'jg': 'juga',
    'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'klw': 'kalau',
    'tp': 'tapi', 'tpi': 'tapi', 'tpi': 'tapi', 'tp': 'tapi',
    'krn': 'karena', 'karna': 'karena', 'krna': 'karena',
    'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah', 'sdh': 'sudah', 'ud': 'sudah',
    'blm': 'belum', 'blum': 'belum', 'blom': 'belum',
    'msh': 'masih', 'msih': 'masih', 'masi': 'masih',
    'sm': 'sama', 'ama': 'sama', 'sma': 'sama',
    'bs': 'bisa', 'bsa': 'bisa', 'bsa': 'bisa',
    'skrg': 'sekarang', 'skrang': 'sekarang', 'skrng': 'sekarang',
    'utk': 'untuk', 'tuk': 'untuk', 'buat': 'untuk', 'bwt': 'untuk',
    'dr': 'dari', 'dri': 'dari',
    'pd': 'pada', 'pda': 'pada',
    'trs': 'terus', 'trus': 'terus', 'trus': 'terus',
    'mulu': 'melulu', 'mlulu': 'melulu', 'mlu': 'melulu',
    'nih': 'ini', 'neh': 'ini', 'ne': 'ini',
    'sih': 'sih',
    'lah': 'lah',
    'dong': 'dong', 'dunk': 'dong', 'donk': 'dong',
    'deh': 'deh',
    'kek': 'seperti', 'kyk': 'seperti', 'kyak': 'seperti',
    'emg': 'memang', 'emang': 'memang', 'mmg': 'memang', 'emng': 'memang',
    'mau': 'mau', 'mo': 'mau',
    'biar': 'supaya', 'byr': 'supaya',
    'dtg': 'datang', 'dtng': 'datang',
    'sdg': 'sedang', 'lg': 'sedang', 'lgi': 'sedang',
    'jd': 'jadi', 'jdi': 'jadi',
    'ttg': 'tentang', 'tntg': 'tentang',
    'hrs': 'harus', 'hrus': 'harus',
    'lbh': 'lebih', 'lebh': 'lebih', 'lbih': 'lebih',
    'dpt': 'dapat', 'dapet': 'dapat', 'dpet': 'dapat',
    'ntar': 'nanti', 'tar': 'nanti', 'ntr': 'nanti',
    'maen': 'main', 'maein': 'main', 'men': 'main',
    'hampir': 'hampir',
    'lagi': 'lagi',
    
    # === PERBAIKAN: Slang yang terlewat dari data asli ===
    'knp': 'kenapa',
    'ngaco': 'kacau',
    'bgs': 'bagus',
    'drak sistem': 'dark sistem',  # typo "derak sistem" → dark system
    'deraksistem': 'dark sistem',
    'dark sistem': 'sistem buruk',
    'lostrek': 'lose streak',
    'anj': '',  # kata kasar → hapus
    'anjir': '',  # kata kasar → hapus
    'anjing': '',  # kata kasar → hapus
    'tai': '',  # kata kasar → hapus
    'bangsat': '',  # kata kasar → hapus
    'tolol': 'bodoh',
    'gblk': 'bodoh',
    'goblog': 'bodoh',
    'bodoh': 'bodoh',
    'bego': 'bodoh',
    'cape': 'capek',
    'capek': 'capek',
    'capee': 'capek',
    'cape': 'capek',
    'anikin': 'menaikkan',
    'naikin': 'menaikkan',
    'kasih': 'kasih',
    'dikasih': 'diberi',
    'ngasih': 'memberi',
    'nurunin': 'menurunkan',
    'turunin': 'menurunkan',
    'ngestak': 'stuck',
    'stuck': 'terjebak',
    'dilasih': 'diberi',  # typo umum
    'pleas': 'tolong',
    'plis': 'tolong',
    'kolep': 'kolaborasi',
    'collab': 'kolaborasi',
    'dhot': 'dot',  # typo umum
    'omg': '',  # ekspresi → hapus
    'fyp': 'trending',
    'orng': 'orang',
    'org': 'orang',
    'reng': 'hero',  # slang MLBB "rank hero"
    
    # === MLBB spesifik ===
    'feeder': 'pemain buruk', 'feeders': 'pemain buruk', 'feed': 'pemain buruk',
    'noob': 'pemain buruk', 'n00b': 'pemain buruk', 'nub': 'pemain buruk', 'nob': 'pemain buruk',
    'afk': 'tidak bermain', 'afkers': 'tidak bermain',
    'bot': 'bot', 'bots': 'bot',
    'matchmaking': 'matchmaking', 'mm': 'matchmaking', 'maching': 'matchmaking',
    'rank': 'peringkat', 'ranked': 'peringkat', 'rangking': 'peringkat',
    'nerf': 'diperlemah', 'buff': 'diperkuat',
    'op': 'terlalu kuat', 'overpowered': 'terlalu kuat', 'imba': 'terlalu kuat',
    'p2w': 'bayar untuk menang', 'pay to win': 'bayar untuk menang',
    'skin': 'kostum', 'skins': 'kostum',
    'diamond': 'berlian', 'dia': 'berlian', 'dimo': 'berlian',
    'hero': 'karakter', 'heroes': 'karakter',
    'push': 'naik rank', 'pushing': 'naik rank',
    'lose streak': 'kalah beruntun', 'losestreak': 'kalah beruntun',
    'win rate': 'tingkat kemenangan', 'winrate': 'tingkat kemenangan', 'wr': 'tingkat kemenangan',
    'lag': 'lag', 'lagg': 'lag', 'lags': 'lag', 'ngelag': 'lag',
    'bug': 'bug', 'buggy': 'bermasalah', 'bugs': 'bug',
    'update': 'pembaruan', 'updates': 'pembaruan',
    'hack': 'curang', 'hacker': 'penipu', 'cheat': 'curang', 'cheater': 'penipu',
    'carry': 'memimpin tim', 'carrier': 'memimpin tim',
    'exp': 'pengalaman', 'experience': 'pengalaman',
    'offlaner': 'penyerang tepi',
    'jungler': 'pemburu hutan', 'jungle': 'hutan',
    'gameplay': 'permainan',
    'graphic': 'grafik', 'graphics': 'grafik', 'grafik': 'grafik',
    'server': 'server', 'servers': 'server',
    'connection': 'koneksi',
    'internet': 'internet',
    'wifi': 'wifi',
    'packet': 'paket data',
    'sinyal': 'sinyal', 'signal': 'sinyal',
    'damage': 'kerusakan', 'dmg': 'kerusakan',
    'cooldown': 'waktu tunggu', 'cd': 'waktu tunggu',
    'mana': 'mana',
    'hp': 'health',
    'skill': 'kemampuan', 'skills': 'kemampuan',
    'ultimate': 'ultimate', 'ult': 'ultimate',
    'tower': 'menara', 'turret': 'menara',
    'lord': 'lord',
    'turtle': 'kura-kura',
    'meta': 'meta',
    'season': 'musim',
    'patch': 'pembaruan',
    'balance': 'keseimbangan', 'balancing': 'keseimbangan',
    'toxic': 'beracun', 'toxicity': 'racun',
    'troll': 'troll', 'trolling': 'troll', 'trollers': 'troll',
    'report': 'laporkan',
    'ban': 'blokir', 'banned': 'diblokir',
    'team': 'tim', 'teams': 'tim', 'teammate': 'teman tim', 'teammates': 'teman tim',
    'enemy': 'musuh', 'enemies': 'musuh',
    'mvp': 'pemain terbaik',
    'bronze': 'perunggu',
    'silver': 'perak',
    'gold': 'emas',
    'epic': 'epic',
    'legend': 'legend',
    'mythic': 'mythic',
    'glory': 'kemuliaan',
    'star': 'bintang', 'stars': 'bintang',
    
    # === Code-mixing (Inggris → Indonesia) ===
    'good': 'bagus', 'great': 'bagus', 'nice': 'bagus', 'best': 'terbaik',
    'bad': 'buruk', 'worst': 'terburuk', 'terrible': 'sangat buruk',
    'fix': 'perbaiki', 'fixing': 'memperbaiki', 'fixed': 'diperbaiki',
    'please': 'tolong', 'pls': 'tolong', 'plz': 'tolong',
    'love': 'suka',
    'hate': 'benci',
    'fun': 'menyenangkan',
    'boring': 'membosankan',
    'waste': 'membuang',
    'support': 'dukungan',
    'random': 'acak',
    'system': 'sistem',
    'player': 'pemain', 'players': 'pemain',
    'unfair': 'tidak adil',
    'fair': 'adil',
    'balance': 'seimbang',
    'unbalance': 'tidak seimbang',
    'crash': 'crash',
    'freeze': 'freeze',
    'error': 'error',
    'problem': 'masalah', 'problems': 'masalah',
    'issue': 'masalah', 'issues': 'masalah',
    
    # === Kata lebay / ekspresi (dihapus) ===
    'wkwk': '', 'wkwkwk': '', 'wkwkwkwk': '', 'wkwkwkwkwk': '',
    'hahaha': '', 'hahah': '', 'haha': '', 'hehe': '', 'hihi': '',
    'lol': '', 'lmao': '', 'lmfao': '', 'rofl': '',
    'fomo': 'takut ketinggalan',
    'ygy': 'ya betul', 'ygw': 'ya sudah',
    'cmn': 'cuman', 'cuma': 'hanya', 'cuman': 'hanya',
    'btw': 'ngomong-ngomong',
    'imo': 'menurut saya', 'imho': 'menurut saya',
    
    # === Typo umum yang terlewat ===
    'bintag': 'bintang', 'bintg': 'bintang',
    'seharusny': 'seharusnya', 'seharusya': 'seharusnya',
    'ngelek': 'lag', 'ngelekin': 'lag', 'ngelag': 'lag',
    'setres': 'stres', 'stress': 'stres', 'stres': 'stres',
    'jelas': 'jelas', 'jelass': 'jelas', 'jelasss': 'jelas',
    'maer': 'mahal', 'mahal': 'mahal',
    'kreeeen': 'keren', 'kereeeen': 'keren', 'kren': 'keren',
    'gara2': 'gara-gara', 'gr2': 'gara-gara',
    'min': 'developer',  # konteks MLBB
    'pihak': 'pihak',
    'kang': 'kakak',
    'bosq': 'bos', 'boss': 'bos',
    'bangus': 'bagus',  # typo dari data
    'bermain': 'bermain',
    'bermasalah': 'bermasalah',
}

# ============================================================
# EMOJI CONVERSION
# ============================================================

EMOJI_TO_TEXT = {
    '😭': ' sangat sedih ', '😢': ' sedih ', '😞': ' kecewa ', '😔': ' kecewa ',
    '😡': ' marah ', '🤬': ' sangat marah ', '😤': ' kesal ', '😠': ' marah ',
    '😒': ' kecewa ', '😩': ' frustasi ', '🙁': ' tidak senang ',
    '😐': ' biasa ', '😑': ' ekspresi datar ',
    '😊': ' senang ', '😄': ' gembira ', '😍': ' sangat suka ', '🥰': ' suka ',
    '😎': ' keren ', '🤩': ' kagum ', '😂': ' lucu ', '🤣': ' lucu ',
    '👍': ' bagus ', '👎': ' buruk ', '✅': ' setuju ', '❌': ' tidak setuju ',
    '🔥': ' keren ', '💔': ' kecewa ', '❤️': ' suka ', '💪': ' kuat ',
    '🙏': ' mohon ', '⭐': ' bintang ', '🌟': ' bintang ',
    '💎': ' berlian ', '🏆': ' juara ', '🚀': ' cepat ', '💸': ' mahal ',
    '🎮': ' game ',
}

# ============================================================
# STOPWORDS CONFIGURATION
# ============================================================

KEEP_WORDS = {
    'bagus', 'baik', 'buruk', 'jelek', 'seru', 'keren', 'bosan',
    'marah', 'kesal', 'senang', 'sedih', 'kecewa', 'stres',
    'frustasi', 'capek', 'lelah', 'puas', 'bangga',
    'lag', 'lambat', 'cepat', 'crash', 'error', 'bug', 'rusak',
    'tidak', 'bukan', 'jangan', 'kurang', 'sangat', 'terlalu',
    'banget', 'sekali', 'lebih', 'paling',
    'menang', 'kalah', 'kuat', 'lemah',
}

MLBB_STOPWORDS = {
    'mobile', 'legends', 'game', 'aplikasi', 'play', 'store',
    'developer', 'moonton', 'mlbb', 'ml',
}

# ============================================================
# INIT LIBRARIES
# ============================================================

# Stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Stopwords
sw_factory = StopWordRemoverFactory()
sastrawi_sw = set(sw_factory.get_stop_words())
nltk_sw = set(stopwords.words('indonesian'))
ALL_STOPWORDS = (sastrawi_sw | nltk_sw | MLBB_STOPWORDS) - KEEP_WORDS

print(f"✅ Pipeline loaded: {len(SLANG_DICT_OPTIMIZED)} slang entries, {len(ALL_STOPWORDS)} stopwords")

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def convert_emoji(text: str) -> str:
    """Konversi emoji ke teks deskriptif."""
    for em, label in EMOJI_TO_TEXT.items():
        text = text.replace(em, label)
    text = emoji.replace_emoji(text, replace=' ')
    return text

def case_fold(text: str) -> str:
    """Lowercase."""
    return text.lower()

def clean_chars(text: str) -> str:
    """Bersihkan URL, HTML, mention, angka, punctuation."""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[!"#$%&\'()*+,\./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7Fa-zA-Z\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_repeating(text: str) -> str:
    """Normalisasi huruf berulang (3+ → 2)."""
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'(\b\w+\b)[\s"\-]+\1', r'\1', text)
    return text

def normalize_slang(text: str) -> str:
    """Normalisasi slang dengan kamus yang diperluas."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        # Check 2-gram
        if i + 1 < len(words):
            bigram = words[i] + ' ' + words[i+1]
            if bigram in SLANG_DICT_OPTIMIZED:
                replacement = SLANG_DICT_OPTIMIZED[bigram]
                if replacement:
                    result.append(replacement)
                i += 2
                continue
        # Check 1-gram
        w = words[i]
        if w in SLANG_DICT_OPTIMIZED:
            replacement = SLANG_DICT_OPTIMIZED[w]
            if replacement:
                result.append(replacement)
        else:
            result.append(w)
        i += 1
    return ' '.join(result)

def tokenize(text: str) -> list:
    """Tokenisasi berbasis spasi."""
    return [w for w in text.split() if len(w) > 1]

def remove_stopwords(tokens: list) -> list:
    """Hapus stopwords."""
    return [t for t in tokens if t not in ALL_STOPWORDS]

def stem_tokens(tokens: list) -> list:
    """Stemming dengan Sastrawi."""
    return [stemmer.stem(t) for t in tokens]

# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_full(text: str, apply_stopword=False, apply_stem=False) -> str:
    """
    Pipeline preprocessing lengkap.
    
    Args:
        text: Teks mentah
        apply_stopword: True → hapus stopwords
        apply_stem: True → lakukan stemming
    
    Returns:
        Teks yang sudah dibersihkan
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    
    text = str(text)
    text = convert_emoji(text)
    text = case_fold(text)
    text = clean_chars(text)
    text = normalize_repeating(text)
    text = normalize_slang(text)
    tokens = tokenize(text)
    
    if apply_stopword:
        tokens = remove_stopwords(tokens)
    if apply_stem:
        tokens = stem_tokens(tokens)
    
    return ' '.join(tokens)

# ============================================================
# LABELING FUNCTION - KONSISTEN
# ============================================================

def rating_to_sentiment(rating):
    """
    Konversi rating ke label sentimen (KONSISTEN).
    
    Rating 1-2 → Negatif
    Rating 3   → Netral
    Rating 4-5 → Positif
    """
    if pd.isna(rating):
        return 'Unknown'
    r = int(rating)
    if r <= 2:
        return 'Negatif'
    elif r == 3:
        return 'Netral'
    else:
        return 'Positif'

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_mlbb_dataset(csv_path: str, output_path: str = None):
    """
    Proses dataset MLBB dengan pipeline yang dioptimasi.
    
    Args:
        csv_path: Path ke file CSV input
        output_path: Path untuk menyimpan hasil (default: auto-generate)
    
    Returns:
        DataFrame yang sudah diproses
    """
    print("="*70)
    print("🚀 MLBB SENTIMENT PREPROCESSING - OPTIMIZED")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"   ✅ Loaded {len(df):,} rows")
    
    # Drop empty reviews
    df = df.dropna(subset=['review']).reset_index(drop=True)
    df = df[df['review'].astype(str).str.strip() != ''].reset_index(drop=True)
    print(f"   ✅ Valid reviews: {len(df):,}")
    
    # PREPROCESSING - 3 VERSI
    print("\n⏳ Processing...")
    print("\n[1/3] Creating review_clean (untuk sentiment analysis, tanpa stopword/stem)...")
    df['review_clean'] = df['review'].progress_apply(
        lambda x: preprocess_full(x, apply_stopword=False, apply_stem=False)
    )
    
    print("\n[2/3] Creating review_stemmed (untuk traditional ML, dengan stemming)...")
    df['review_stemmed'] = df['review'].progress_apply(
        lambda x: preprocess_full(x, apply_stopword=False, apply_stem=True)
    )
    
    print("\n[3/3] Creating review_topic (untuk topic modeling, stopword + stem)...")
    df['review_topic'] = df['review'].progress_apply(
        lambda x: preprocess_full(x, apply_stopword=True, apply_stem=True)
    )
    
    # Add statistics
    df['review_clean_length'] = df['review_clean'].apply(len)
    df['review_clean_words'] = df['review_clean'].apply(lambda x: len(x.split()))
    
    # Drop empty after preprocessing
    df_empty = df[df['review_clean'].str.strip() == '']
    if len(df_empty) > 0:
        print(f"\n⚠️  {len(df_empty)} reviews became empty after preprocessing → removed")
        df = df[df['review_clean'].str.strip() != ''].reset_index(drop=True)
    
    # LABELING - KONSISTEN
    if 'rating' in df.columns:
        print("\n🏷️  Creating consistent sentiment labels...")
        df['sentiment_label'] = df['rating'].apply(rating_to_sentiment)
        df['sentiment_id'] = df['sentiment_label'].map({
            'Negatif': 0, 
            'Netral': 1, 
            'Positif': 2, 
            'Unknown': -1
        })
        
        # Show distribution
        print("\n📊 Sentiment Distribution:")
        dist = df['sentiment_label'].value_counts()
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"   {label:8s}: {count:5,} ({pct:5.2f}%)")
    
    # Save output
    if output_path is None:
        output_path = csv_path.replace('.csv', '_optimized.csv')
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Processing complete!")
    print(f"   Output saved to: {output_path}")
    print(f"   Final rows: {len(df):,}")
    
    # Show sample
    print("\n📋 Sample output (first 3 rows):")
    sample_cols = ['review', 'review_clean', 'review_stemmed', 'sentiment_label']
    available_cols = [c for c in sample_cols if c in df.columns]
    print(df[available_cols].head(3).to_string(index=False))
    
    return df

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Quick test dengan sample text
    print("\n" + "="*70)
    print("🧪 QUICK TEST")
    print("="*70)
    
    test_cases = [
        'plis lah moonton saya cape anikin reng tapi knp di kasih tim bot semua',
        'developer kurangi deraksistem nya bikin stres masa miya pengalaman',
        'moonton ngestak di epic saya naikin lah dilasih dark sistem melulu saya',
        'bagusssss banget gamenya suka bangettt',
        'game ngaco anj bgs banget lostrek melulu',
    ]
    
    for i, text in enumerate(test_cases, 1):
        clean = preprocess_full(text, apply_stopword=False, apply_stem=False)
        stemmed = preprocess_full(text, apply_stopword=False, apply_stem=True)
        print(f"\n[{i}] ORIGINAL: {text}")
        print(f"    CLEAN   : {clean}")
        print(f"    STEMMED : {stemmed}")
    
    print("\n" + "="*70)
    print("✅ Script ready! Use: process_mlbb_dataset('your_file.csv')")
    print("="*70)