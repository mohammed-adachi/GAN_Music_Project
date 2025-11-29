import requests
import os
import time
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION (Inchangée)
# ==============================================================================

SAVE_DIR = "bitmidi_dataset"
BASE_URL = "https://bitmidi.com"
SITEMAP_INDEX_URL = "https://bitmidi.com/sitemap.xml" # Renommé pour plus de clarté
DELAY_SECONDS = 2
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==============================================================================
# 2. LE SCRIPT DE SCRAPING (MIS À JOUR)
# ==============================================================================

def scrape_bitmidi():
    print("🚀 Démarrage du scraping de bitmidi.com...")
    print(f"Les fichiers seront sauvegardés dans le dossier : '{SAVE_DIR}'")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- Étape 1: Gérer l'index de sitemaps ---
    print(f"\n1. Téléchargement de l'index de sitemaps depuis {SITEMAP_INDEX_URL}...")
    try:
        response = requests.get(SITEMAP_INDEX_URL, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ ERREUR : Impossible de télécharger l'index de sitemaps. Erreur : {e}")
        return

    # Analyser l'index pour trouver les URLs des VRAIES sitemaps
    root = ET.fromstring(response.content)
    ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    # Le chemin a changé : on cherche <sitemap> puis <loc>
    sitemap_urls = [elem.text for elem in root.findall('sitemap:sitemap/sitemap:loc', ns)]
    
    if not sitemap_urls:
        print("❌ ERREUR : Aucune sitemap trouvée dans l'index. Le site a peut-être encore changé de structure.")
        return
        
    print(f"✅ {len(sitemap_urls)} sitemap(s) trouvé(s) dans l'index. Analyse de chaque sitemap...")
    
    # --- Étape 1.5: Parcourir chaque sitemap pour collecter TOUTES les URLs ---
    all_song_urls = []
    for sitemap_url in tqdm(sitemap_urls, desc="Analyse des sitemaps"):
        try:
            time.sleep(1) # Petit délai
            sitemap_response = requests.get(sitemap_url, headers=HEADERS)
            sitemap_response.raise_for_status()
            
            sitemap_root = ET.fromstring(sitemap_response.content)
            # Ici on utilise l'ancien chemin pour trouver les URLs des chansons
            urls_from_this_map = [elem.text for elem in sitemap_root.findall('sitemap:url/sitemap:loc', ns)]
            all_song_urls.extend(urls_from_this_map)
            
        except requests.exceptions.RequestException as e:
            tqdm.write(f"⚠️ Impossible de traiter la sitemap {sitemap_url}: {e}")

    # Filtrer la liste complète pour ne garder que les pages de chansons
    midi_page_urls = [url for url in all_song_urls if url.endswith('-mid')]
    
    if not midi_page_urls:
        print("❌ ERREUR : Aucune URL de chanson trouvée après avoir analysé toutes les sitemaps.")
        return
        
    print(f"✅ Au total, {len(midi_page_urls)} pages de chansons ont été collectées.")
    
    # --- Étape 2: Démarrer le téléchargement des fichiers (inchangée) ---
    print("\n2. Démarrage du téléchargement des fichiers MIDI (cela peut prendre plusieurs heures)...")
    
    for url in tqdm(midi_page_urls, desc="Téléchargement", unit="fichier"):
        try:
            time.sleep(DELAY_SECONDS)
            page_response = requests.get(url, headers=HEADERS)
            page_response.raise_for_status()
            
            soup = BeautifulSoup(page_response.content, 'html.parser')
            download_link_tag = soup.find('a', attrs={'download': True})
            
            if download_link_tag and 'href' in download_link_tag.attrs:
                relative_link = download_link_tag['href']
                download_url = BASE_URL + relative_link
                filename = download_link_tag['download']
                if not filename.lower().endswith('.mid'):
                    filename += '.mid'
                save_path = os.path.join(SAVE_DIR, filename)

                if not os.path.exists(save_path):
                    midi_response = requests.get(download_url, headers=HEADERS)
                    midi_response.raise_for_status()
                    with open(save_path, 'wb') as f:
                        f.write(midi_response.content)
                        
        except requests.exceptions.RequestException as e:
            tqdm.write(f"⚠️ Erreur de réseau pour {url}: {e}")
        except Exception as e:
            tqdm.write(f"⚠️ Erreur inattendue pour {url}: {e}")

    print("\n🎉 Scraping terminé !")
    print(f"Tous les fichiers disponibles ont été téléchargés dans le dossier '{SAVE_DIR}'.")

if __name__ == '__main__':
    scrape_bitmidi()