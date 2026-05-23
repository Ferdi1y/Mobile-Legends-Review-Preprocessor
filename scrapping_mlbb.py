"""
Scraping Review Mobile Legends: Bang Bang dari Google Play Store
Author: Data Science Student
Date: 2024
"""

from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime
import time

def scrape_mlbb_reviews(target_count=10000):
    """
    Melakukan scraping review aplikasi Mobile Legends dari Google Play Store
    
    Parameters:
    -----------
    target_count : int
        Jumlah target review yang ingin diambil (minimal 10000)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi review yang telah di-scraping
    """
    
    print("=" * 60)
    print("SCRAPING REVIEW MOBILE LEGENDS: BANG BANG")
    print("=" * 60)
    print(f"Target review: {target_count}")
    print(f"Aplikasi: com.mobile.legends")
    print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    app_id = 'com.mobile.legends'
    all_reviews = []
    continuation_token = None
    
    # Scraping dengan pagination
    batch_num = 1
    while len(all_reviews) < target_count:
        try:
            print(f"\n[Batch {batch_num}] Mengambil review...")
            
            # Ambil review
            result, continuation_token = reviews(
                app_id,
                lang='id',  # Bahasa Indonesia
                country='id',  # Indonesia
                sort=Sort.NEWEST,  # Urutkan dari yang terbaru
                count=200,  # Ambil 200 review per batch
                continuation_token=continuation_token
            )
            
            all_reviews.extend(result)
            print(f"[Batch {batch_num}] Berhasil mengambil {len(result)} review")
            print(f"Total review terkumpul: {len(all_reviews)}/{target_count}")
            
            # Cek apakah sudah mencapai target
            if len(all_reviews) >= target_count:
                print(f"\n✓ Target tercapai! Total review: {len(all_reviews)}")
                break
            
            # Cek apakah masih ada continuation token
            if continuation_token is None:
                print("\n⚠ Tidak ada lagi review yang tersedia.")
                break
            
            batch_num += 1
            
            # Delay untuk menghindari rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"\n✗ Error pada batch {batch_num}: {str(e)}")
            print("Mencoba melanjutkan...")
            time.sleep(3)
            continue
    
    # Konversi ke DataFrame
    print("\n" + "=" * 60)
    print("MEMPROSES DATA...")
    print("=" * 60)
    
    df = pd.DataFrame(all_reviews)
    
    # Pilih kolom yang relevan
    df_clean = df[['userName', 'score', 'content', 'at', 'reviewCreatedVersion', 
                   'thumbsUpCount', 'replyContent']].copy()
    
    # Rename kolom agar lebih mudah dipahami
    df_clean.columns = ['nama_pengguna', 'rating', 'review', 'tanggal', 
                        'versi_app', 'jumlah_like', 'balasan_developer']
    
    # Konversi tanggal
    df_clean['tanggal'] = pd.to_datetime(df_clean['tanggal'])
    
    # Remove duplikat
    df_clean = df_clean.drop_duplicates(subset=['review'], keep='first')
    
    # Remove review kosong
    df_clean = df_clean[df_clean['review'].notna()]
    df_clean = df_clean[df_clean['review'].str.strip() != '']
    
    print(f"\nTotal review setelah pembersihan: {len(df_clean)}")
    print(f"Rentang tanggal: {df_clean['tanggal'].min()} - {df_clean['tanggal'].max()}")
    print(f"\nDistribusi Rating:")
    print(df_clean['rating'].value_counts().sort_index())
    
    return df_clean


def save_to_csv(df, filename='mlbb_reviews.csv'):
    """
    Menyimpan DataFrame ke file CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan disimpan
    filename : str
        Nama file output
    """
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ Data berhasil disimpan ke: {filename}")
    print(f"✓ Total baris: {len(df)}")
    print(f"✓ Total kolom: {len(df.columns)}")


if __name__ == "__main__":
    # Scraping review
    df_reviews = scrape_mlbb_reviews(target_count=10000)
    
    # Simpan ke CSV
    save_to_csv(df_reviews, 'mlbb_reviews.csv')
    
    print("\n" + "=" * 60)
    print("SCRAPING SELESAI!")
    print("=" * 60)
    print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")