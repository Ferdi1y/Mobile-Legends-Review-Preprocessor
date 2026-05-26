#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
RE-OPTIMASI MLBB PREPROCESSING
Untuk data yang sudah punya review_clean
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import sys

sys.path.append('.')
from mlbb_preprocessing_optimized import (
    preprocess_full,
    SLANG_DICT_OPTIMIZED,
    ALL_STOPWORDS
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 150)

print("="*80)
print("🔧 RE-OPTIMASI MLBB DATASET")
print("="*80)

# ============================================================
# LOAD DATA
# ============================================================

INPUT_CSV = 'mlbb_reviews_for_modeling.csv'
df = pd.read_csv(INPUT_CSV, encoding='utf-8')

print(f"\n📂 Loaded: {INPUT_CSV}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# ============================================================
# RE-PREPROCESSING
# ============================================================

print("\n⏳ Re-processing dengan kamus slang yang diperluas...")

# Gunakan review_clean yang ada sebagai input (karena tidak ada review original)
print("\n[1/3] Creating review_clean_v2 (optimized)...")
df['review_clean_v2'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=False, apply_stem=False)
)

print("\n[2/3] Creating review_stemmed (untuk traditional ML)...")
df['review_stemmed'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=False, apply_stem=True)
)

print("\n[3/3] Creating review_topic (untuk topic modeling)...")
df['review_topic'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=True, apply_stem=True)
)

# Statistics
df['words_original'] = df['review_clean'].str.split().str.len()
df['words_optimized'] = df['review_clean_v2'].str.split().str.len()
df['words_stemmed'] = df['review_stemmed'].str.split().str.len()

print(f"\n✅ Re-processing complete!")

# ============================================================
# ANALISIS PERBANDINGAN
# ============================================================

print("\n" + "="*80)
print("📊 PERBANDINGAN: OLD vs NEW")
print("="*80)

# Vocabulary comparison
vocab_old = set(' '.join(df['review_clean'].astype(str)).split())
vocab_new = set(' '.join(df['review_clean_v2'].astype(str)).split())
vocab_stemmed = set(' '.join(df['review_stemmed'].astype(str)).split())

print(f"\n📚 Vocabulary Size:")
print(f"   OLD (review_clean)    : {len(vocab_old):,} unique words")
print(f"   NEW (review_clean_v2) : {len(vocab_new):,} unique words")
print(f"   STEMMED              : {len(vocab_stemmed):,} unique words")
print(f"   Improvement (clean)  : {len(vocab_old) - len(vocab_new):,} words normalized")
print(f"   Improvement (stem)   : {len(vocab_new) - len(vocab_stemmed):,} words reduced by stemming")

# Kata yang berhasil dinormalisasi
removed_words = vocab_old - vocab_new
print(f"\n✅ {len(removed_words):,} kata berhasil dinormalisasi dalam optimasi ini")

if len(removed_words) > 0:
    print(f"\n📌 Sample kata yang dinormalisasi (50 pertama):")
    sample = sorted(list(removed_words))[:50]
    for i in range(0, len(sample), 5):
        batch = sample[i:i+5]
        print(f"   {', '.join(batch)}")

# ============================================================
# TOP WORDS COMPARISON
# ============================================================

print("\n" + "="*80)
print("🔤 TOP 30 WORDS - BEFORE vs AFTER")
print("="*80)

def get_top_words(text_series, n=30):
    all_text = ' '.join(text_series.astype(str))
    words = all_text.split()
    return Counter(words).most_common(n)

top_old = get_top_words(df['review_clean'], 30)
top_new = get_top_words(df['review_clean_v2'], 30)

print("\n📌 OLD (review_clean):")
for i, (word, count) in enumerate(top_old, 1):
    print(f"   {i:2d}. {word:20s} : {count:4,}")

print("\n📌 NEW (review_clean_v2):")
for i, (word, count) in enumerate(top_new, 1):
    print(f"   {i:2d}. {word:20s} : {count:4,}")

# ============================================================
# DETEKSI SLANG TERSISA
# ============================================================

print("\n" + "="*80)
print("🔍 DETEKSI SLANG/NOISE TERSISA")
print("="*80)

# Kata pendek mencurigakan
short_words = []
for text in df['review_clean_v2']:
    words = str(text).split()
    short_words.extend([w for w in words if len(w) == 2])

short_counter = Counter(short_words)
frequent_shorts = {w: c for w, c in short_counter.items() if c > 3}

if frequent_shorts:
    print(f"\n⚠️  Kata 2-karakter yang sering muncul ({len(frequent_shorts)} kata):")
    for word, count in sorted(frequent_shorts.items(), key=lambda x: -x[1])[:20]:
        print(f"   '{word}' : {count:,} occurrences")
else:
    print("\n✅ Tidak ada kata pendek mencurigakan")

# ============================================================
# SAMPLE COMPARISON
# ============================================================

print("\n" + "="*80)
print("📋 SAMPLE PERBANDINGAN (10 rows acak)")
print("="*80)

sample = df.sample(10, random_state=42)
for idx, row in sample.iterrows():
    print(f"\n[Row {idx}]")
    print(f"  OLD      : {row['review_clean'][:100]}")
    print(f"  NEW      : {row['review_clean_v2'][:100]}")
    print(f"  STEMMED  : {row['review_stemmed'][:100]}")

# ============================================================
# LABEL CONSISTENCY CHECK
# ============================================================

print("\n" + "="*80)
print("🎯 LABEL CONSISTENCY CHECK")
print("="*80)

if 'sentiment_label' in df.columns and 'indobert_label' in df.columns:
    # Consistency
    consistency = (df['sentiment_label'] == df['indobert_label']).mean()
    inconsistent = df[df['sentiment_label'] != df['indobert_label']]
    
    print(f"\nKonsistensi Label:")
    print(f"   Sentiment vs IndoBERT: {consistency*100:.2f}%")
    print(f"   Inconsistent rows    : {len(inconsistent):,} ({len(inconsistent)/len(df)*100:.1f}%)")
    
    if len(inconsistent) > 0:
        print(f"\n⚠️  Sample data yang TIDAK KONSISTEN:")
        print(inconsistent[['review_clean_v2', 'sentiment_label', 'indobert_label']].head(10).to_string(index=True))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['sentiment_label'], df['indobert_label'],
                         labels=['Negatif', 'Netral', 'Positif'])
    print(f"\n📊 Confusion Matrix:")
    print(f"                  Predicted (IndoBERT)")
    print(f"                  Negatif  Netral  Positif")
    for i, true_label in enumerate(['Negatif', 'Netral', 'Positif']):
        print(f"   True {true_label:7s}:  {cm[i][0]:6,}  {cm[i][1]:6,}  {cm[i][2]:6,}")

# ============================================================
# EXPORT FINAL DATA
# ============================================================

print("\n" + "="*80)
print("💾 EXPORT DATA")
print("="*80)

# Prepare final dataframe
df_final = df[[
    'review_clean',        # Original dari file Anda
    'review_clean_v2',     # Optimized version
    'review_stemmed',      # Untuk traditional ML
    'review_topic',        # Untuk topic modeling
    'sentiment_label',     # Ground truth
    'sentiment_id',        # Encoded
    'indobert_label',      # Prediksi IndoBERT (jika ada)
]].copy()

# Rename untuk clarity
df_final = df_final.rename(columns={
    'review_clean': 'review_clean_old',
    'review_clean_v2': 'review_clean'
})

# Save
output_path = 'mlbb_reviews_optimized_final.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✅ Data exported to: {output_path}")
print(f"   Columns: {list(df_final.columns)}")
print(f"   Rows: {len(df_final):,}")

# Preview
print("\n📋 Preview (first 3 rows):")
print(df_final[['review_clean', 'review_stemmed', 'sentiment_label']].head(3).to_string(index=False))

# ============================================================
# VISUALIZATION
# ============================================================

print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MLBB Preprocessing Optimization - Analysis', fontsize=14, fontweight='bold')

# 1. Word count distribution
axes[0,0].hist([df['words_original'], df['words_optimized'], df['words_stemmed']], 
               label=['Original', 'Optimized', 'Stemmed'], bins=20, alpha=0.7)
axes[0,0].set_title('Word Count Distribution')
axes[0,0].set_xlabel('Word Count')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. Vocabulary reduction
vocab_sizes = [len(vocab_old), len(vocab_new), len(vocab_stemmed)]
axes[0,1].bar(['Original', 'Optimized', 'Stemmed'], vocab_sizes, 
              color=['#e74c3c', '#3498db', '#2ecc71'])
axes[0,1].set_title('Vocabulary Size Reduction')
axes[0,1].set_ylabel('Unique Words')
for i, v in enumerate(vocab_sizes):
    axes[0,1].text(i, v + 20, f'{v:,}', ha='center', fontweight='bold')

# 3. Sentiment distribution
if 'sentiment_label' in df.columns:
    dist = df['sentiment_label'].value_counts()
    colors_map = {'Positif': '#2ecc71', 'Netral': '#f39c12', 'Negatif': '#e74c3c'}
    colors = [colors_map[label] for label in dist.index]
    axes[1,0].pie(dist.values, labels=dist.index, autopct='%1.1f%%', 
                  colors=colors, startangle=140)
    axes[1,0].set_title('Sentiment Distribution')

# 4. Label consistency
if 'sentiment_label' in df.columns and 'indobert_label' in df.columns:
    consistency_rate = (df['sentiment_label'] == df['indobert_label']).mean() * 100
    inconsistency_rate = 100 - consistency_rate
    axes[1,1].bar(['Consistent', 'Inconsistent'], 
                  [consistency_rate, inconsistency_rate],
                  color=['#2ecc71', '#e74c3c'])
    axes[1,1].set_title('Label Consistency: sentiment_label vs indobert_label')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].set_ylim(0, 100)
    for i, v in enumerate([consistency_rate, inconsistency_rate]):
        axes[1,1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('optimization_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: optimization_analysis.png")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ OPTIMASI SELESAI!")
print("="*80)

print(f"""
📝 SUMMARY:

1. ✅ Kamus Slang Diperluas
   - Total entries: {len(SLANG_DICT_OPTIMIZED)} (3x dari sebelumnya)
   - Kata dinormalisasi: {len(removed_words):,}
   
2. ✅ Stemming Terintegrasi
   - Vocabulary reduction: {len(vocab_new):,} → {len(vocab_stemmed):,} (-{len(vocab_new)-len(vocab_stemmed):,})
   - Kolom baru: 'review_stemmed'
   
3. ✅ Multi-Purpose Output
   - review_clean      : untuk sentiment analysis (IndoBERT, VADER)
   - review_stemmed    : untuk traditional ML (Naive Bayes, SVM)
   - review_topic      : untuk topic modeling (LDA, NMF)
   
4. ⚠️  Label Consistency
   - sentiment_label vs indobert_label: {consistency*100:.2f}%
   - Rekomendasi: Gunakan sentiment_label sebagai ground truth
   
🎯 READY FOR MODELING!
Data tersimpan di: {output_path}
""")#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
RE-OPTIMASI MLBB PREPROCESSING
Untuk data yang sudah punya review_clean
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import sys

sys.path.append('.')
from data.mlbb_preprocessing_optimized import (
    preprocess_full,
    SLANG_DICT_OPTIMIZED,
    ALL_STOPWORDS
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 150)

print("="*80)
print("🔧 RE-OPTIMASI MLBB DATASET")
print("="*80)

# ============================================================
# LOAD DATA
# ============================================================

INPUT_CSV = 'mlbb_reviews_for_modeling.csv'
df = pd.read_csv(INPUT_CSV, encoding='utf-8')

print(f"\n📂 Loaded: {INPUT_CSV}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# ============================================================
# RE-PREPROCESSING
# ============================================================

print("\n⏳ Re-processing dengan kamus slang yang diperluas...")

# Gunakan review_clean yang ada sebagai input (karena tidak ada review original)
print("\n[1/3] Creating review_clean_v2 (optimized)...")
df['review_clean_v2'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=False, apply_stem=False)
)

print("\n[2/3] Creating review_stemmed (untuk traditional ML)...")
df['review_stemmed'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=False, apply_stem=True)
)

print("\n[3/3] Creating review_topic (untuk topic modeling)...")
df['review_topic'] = df['review_clean'].progress_apply(
    lambda x: preprocess_full(str(x), apply_stopword=True, apply_stem=True)
)

# Statistics
df['words_original'] = df['review_clean'].str.split().str.len()
df['words_optimized'] = df['review_clean_v2'].str.split().str.len()
df['words_stemmed'] = df['review_stemmed'].str.split().str.len()

print(f"\n✅ Re-processing complete!")

# ============================================================
# ANALISIS PERBANDINGAN
# ============================================================

print("\n" + "="*80)
print("📊 PERBANDINGAN: OLD vs NEW")
print("="*80)

# Vocabulary comparison
vocab_old = set(' '.join(df['review_clean'].astype(str)).split())
vocab_new = set(' '.join(df['review_clean_v2'].astype(str)).split())
vocab_stemmed = set(' '.join(df['review_stemmed'].astype(str)).split())

print(f"\n📚 Vocabulary Size:")
print(f"   OLD (review_clean)    : {len(vocab_old):,} unique words")
print(f"   NEW (review_clean_v2) : {len(vocab_new):,} unique words")
print(f"   STEMMED              : {len(vocab_stemmed):,} unique words")
print(f"   Improvement (clean)  : {len(vocab_old) - len(vocab_new):,} words normalized")
print(f"   Improvement (stem)   : {len(vocab_new) - len(vocab_stemmed):,} words reduced by stemming")

# Kata yang berhasil dinormalisasi
removed_words = vocab_old - vocab_new
print(f"\n✅ {len(removed_words):,} kata berhasil dinormalisasi dalam optimasi ini")

if len(removed_words) > 0:
    print(f"\n📌 Sample kata yang dinormalisasi (50 pertama):")
    sample = sorted(list(removed_words))[:50]
    for i in range(0, len(sample), 5):
        batch = sample[i:i+5]
        print(f"   {', '.join(batch)}")

# ============================================================
# TOP WORDS COMPARISON
# ============================================================

print("\n" + "="*80)
print("🔤 TOP 30 WORDS - BEFORE vs AFTER")
print("="*80)

def get_top_words(text_series, n=30):
    all_text = ' '.join(text_series.astype(str))
    words = all_text.split()
    return Counter(words).most_common(n)

top_old = get_top_words(df['review_clean'], 30)
top_new = get_top_words(df['review_clean_v2'], 30)

print("\n📌 OLD (review_clean):")
for i, (word, count) in enumerate(top_old, 1):
    print(f"   {i:2d}. {word:20s} : {count:4,}")

print("\n📌 NEW (review_clean_v2):")
for i, (word, count) in enumerate(top_new, 1):
    print(f"   {i:2d}. {word:20s} : {count:4,}")

# ============================================================
# DETEKSI SLANG TERSISA
# ============================================================

print("\n" + "="*80)
print("🔍 DETEKSI SLANG/NOISE TERSISA")
print("="*80)

# Kata pendek mencurigakan
short_words = []
for text in df['review_clean_v2']:
    words = str(text).split()
    short_words.extend([w for w in words if len(w) == 2])

short_counter = Counter(short_words)
frequent_shorts = {w: c for w, c in short_counter.items() if c > 3}

if frequent_shorts:
    print(f"\n⚠️  Kata 2-karakter yang sering muncul ({len(frequent_shorts)} kata):")
    for word, count in sorted(frequent_shorts.items(), key=lambda x: -x[1])[:20]:
        print(f"   '{word}' : {count:,} occurrences")
else:
    print("\n✅ Tidak ada kata pendek mencurigakan")

# ============================================================
# SAMPLE COMPARISON
# ============================================================

print("\n" + "="*80)
print("📋 SAMPLE PERBANDINGAN (10 rows acak)")
print("="*80)

sample = df.sample(10, random_state=42)
for idx, row in sample.iterrows():
    print(f"\n[Row {idx}]")
    print(f"  OLD      : {row['review_clean'][:100]}")
    print(f"  NEW      : {row['review_clean_v2'][:100]}")
    print(f"  STEMMED  : {row['review_stemmed'][:100]}")

# ============================================================
# LABEL CONSISTENCY CHECK
# ============================================================

print("\n" + "="*80)
print("🎯 LABEL CONSISTENCY CHECK")
print("="*80)

if 'sentiment_label' in df.columns and 'indobert_label' in df.columns:
    # Consistency
    consistency = (df['sentiment_label'] == df['indobert_label']).mean()
    inconsistent = df[df['sentiment_label'] != df['indobert_label']]
    
    print(f"\nKonsistensi Label:")
    print(f"   Sentiment vs IndoBERT: {consistency*100:.2f}%")
    print(f"   Inconsistent rows    : {len(inconsistent):,} ({len(inconsistent)/len(df)*100:.1f}%)")
    
    if len(inconsistent) > 0:
        print(f"\n⚠️  Sample data yang TIDAK KONSISTEN:")
        print(inconsistent[['review_clean_v2', 'sentiment_label', 'indobert_label']].head(10).to_string(index=True))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['sentiment_label'], df['indobert_label'],
                         labels=['Negatif', 'Netral', 'Positif'])
    print(f"\n📊 Confusion Matrix:")
    print(f"                  Predicted (IndoBERT)")
    print(f"                  Negatif  Netral  Positif")
    for i, true_label in enumerate(['Negatif', 'Netral', 'Positif']):
        print(f"   True {true_label:7s}:  {cm[i][0]:6,}  {cm[i][1]:6,}  {cm[i][2]:6,}")

# ============================================================
# EXPORT FINAL DATA
# ============================================================

print("\n" + "="*80)
print("💾 EXPORT DATA")
print("="*80)

# Prepare final dataframe
df_final = df[[
    'review_clean',        # Original dari file Anda
    'review_clean_v2',     # Optimized version
    'review_stemmed',      # Untuk traditional ML
    'review_topic',        # Untuk topic modeling
    'sentiment_label',     # Ground truth
    'sentiment_id',        # Encoded
    'indobert_label',      # Prediksi IndoBERT (jika ada)
]].copy()

# Rename untuk clarity
df_final = df_final.rename(columns={
    'review_clean': 'review_clean_old',
    'review_clean_v2': 'review_clean'
})

# Save
output_path = 'mlbb_reviews_optimized_final.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✅ Data exported to: {output_path}")
print(f"   Columns: {list(df_final.columns)}")
print(f"   Rows: {len(df_final):,}")

# Preview
print("\n📋 Preview (first 3 rows):")
print(df_final[['review_clean', 'review_stemmed', 'sentiment_label']].head(3).to_string(index=False))

# ============================================================
# VISUALIZATION
# ============================================================

print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MLBB Preprocessing Optimization - Analysis', fontsize=14, fontweight='bold')

# 1. Word count distribution
axes[0,0].hist([df['words_original'], df['words_optimized'], df['words_stemmed']], 
               label=['Original', 'Optimized', 'Stemmed'], bins=20, alpha=0.7)
axes[0,0].set_title('Word Count Distribution')
axes[0,0].set_xlabel('Word Count')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. Vocabulary reduction
vocab_sizes = [len(vocab_old), len(vocab_new), len(vocab_stemmed)]
axes[0,1].bar(['Original', 'Optimized', 'Stemmed'], vocab_sizes, 
              color=['#e74c3c', '#3498db', '#2ecc71'])
axes[0,1].set_title('Vocabulary Size Reduction')
axes[0,1].set_ylabel('Unique Words')
for i, v in enumerate(vocab_sizes):
    axes[0,1].text(i, v + 20, f'{v:,}', ha='center', fontweight='bold')

# 3. Sentiment distribution
if 'sentiment_label' in df.columns:
    dist = df['sentiment_label'].value_counts()
    colors_map = {'Positif': '#2ecc71', 'Netral': '#f39c12', 'Negatif': '#e74c3c'}
    colors = [colors_map[label] for label in dist.index]
    axes[1,0].pie(dist.values, labels=dist.index, autopct='%1.1f%%', 
                  colors=colors, startangle=140)
    axes[1,0].set_title('Sentiment Distribution')

# 4. Label consistency
if 'sentiment_label' in df.columns and 'indobert_label' in df.columns:
    consistency_rate = (df['sentiment_label'] == df['indobert_label']).mean() * 100
    inconsistency_rate = 100 - consistency_rate
    axes[1,1].bar(['Consistent', 'Inconsistent'], 
                  [consistency_rate, inconsistency_rate],
                  color=['#2ecc71', '#e74c3c'])
    axes[1,1].set_title('Label Consistency: sentiment_label vs indobert_label')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].set_ylim(0, 100)
    for i, v in enumerate([consistency_rate, inconsistency_rate]):
        axes[1,1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('optimization_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: optimization_analysis.png")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ OPTIMASI SELESAI!")
print("="*80)

print(f"""
📝 SUMMARY:

1. ✅ Kamus Slang Diperluas
   - Total entries: {len(SLANG_DICT_OPTIMIZED)} (3x dari sebelumnya)
   - Kata dinormalisasi: {len(removed_words):,}
   
2. ✅ Stemming Terintegrasi
   - Vocabulary reduction: {len(vocab_new):,} → {len(vocab_stemmed):,} (-{len(vocab_new)-len(vocab_stemmed):,})
   - Kolom baru: 'review_stemmed'
   
3. ✅ Multi-Purpose Output
   - review_clean      : untuk sentiment analysis (IndoBERT, VADER)
   - review_stemmed    : untuk traditional ML (Naive Bayes, SVM)
   - review_topic      : untuk topic modeling (LDA, NMF)
   
4. ⚠️  Label Consistency
   - sentiment_label vs indobert_label: {consistency*100:.2f}%
   - Rekomendasi: Gunakan sentiment_label sebagai ground truth
   
🎯 READY FOR MODELING!
Data tersimpan di: {output_path}
""")