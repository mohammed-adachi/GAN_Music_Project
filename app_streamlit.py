# ==============================================================================
# APPLICATION STREAMLIT - APPASSIONATO (MUSEGAN)
# ==============================================================================
# PRÉREQUIS (À lancer dans le terminal avant) :
# sudo apt-get install -y fluidsynth
# pip install midi2audio streamlit pypianoroll torch
# ==============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pypianoroll
import subprocess

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="APPASSIONATO - IA Compositrice",
    page_icon="🎼",
    layout="wide"
)

# ==============================================================================
# 2. PARAMÈTRES ET CONFIGURATION
# ==============================================================================
# Configuration système
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "musegan_checkpoint_4.pth"

# Paramètres du modèle (Ne pas toucher, doit correspondre à l'entraînement)
latent_dim = 128
n_tracks = 5
n_measures = 4
measure_resolution = 16
n_pitches = 72
lowest_pitch = 24

# Paramètres MIDI
EXPORT_RESOLUTION = 24
UPSCALE_FACTOR = EXPORT_RESOLUTION // 4  # Passage de res 4 à 24

# --- SIDEBAR : CONTRÔLES ET INSTRUMENTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1207/1207249.png", width=100)
    st.title("🎛️ Studio IA")
    
    st.markdown("### 1. Ambiance Sonore")
    style_choice = st.selectbox(
        "Choisir les instruments :",
        ("Classique (Piano & Cordes)", "Acoustique (Guitares)", "Électronique (Synthé)")
    )

    # Configuration des instruments selon le choix (Standard General MIDI)
    if style_choice == "Classique (Piano & Cordes)":
        track_names = ["Piano Principal", "Piano Accomp.", "Basse", "Ensemble Cordes", "Batterie"]
        programs = [0, 0, 33, 48, 0] # 0=Piano, 33=Bass, 48=Strings
        tempo = 65
        
    elif style_choice == "Acoustique (Guitares)":
        track_names = ["Guitare Folk", "Guitare Nylon", "Basse Acoustique", "Violon", "Percussions"]
        programs = [25, 24, 32, 40, 0] # 25=Steel Gtr, 24=Nylon Gtr, 40=Violin
        tempo = 70

    elif style_choice == "Électronique (Synthé)":
        track_names = ["Clavier Rhodes", "Synthé Pad", "Synth Bass", "Lead (Sawtooth)", "Drum Kit"]
        programs = [4, 89, 38, 81, 0] # 4=Rhodes, 89=Warm Pad, 38=Synth Bass
        tempo = 80

    # Indicateur batterie (Toujours le dernier track dans notre modèle)
    is_drums = [False, False, False, False, True]

    st.markdown("### 2. Paramètres de génération")
    nb_samples = st.slider("Durée (Nombre de blocs)", 1, 4, 2, help="1 bloc = 4 mesures")
    
    st.markdown("---")
    st.info(f"**Modèle chargé :** {style_choice}")
    st.write("**Réalisé par :**\nMohammed ADACHI\nOthmane LAKLECH")

# ==============================================================================
# 3. ARCHITECTURE DU MODÈLE (GÉNÉRATEUR)
# ==============================================================================
class GeneraterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class Generator(nn.Module):
    def __init__(self, z_dim=latent_dim, n_tracks=n_tracks):
        super().__init__()
        self.z_dim = z_dim
        # Tronc Commun
        self.transconv0 = GeneraterBlock(self.z_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        # Branches
        self.transconv4 = nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])
        self.transconv5 = nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        track_outputs = [transconv(x) for transconv in self.transconv4]
        track_outputs = torch.cat([transconv(track) for track, transconv in zip(track_outputs, self.transconv5)], 1)
        track_outputs = track_outputs.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return torch.sigmoid(track_outputs)

# ==============================================================================
# 4. FONCTIONS UTILITAIRES
# ==============================================================================
@st.cache_resource
def load_model():
    """Charge le modèle et les poids"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    model = Generator().to(device)
    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur poids : {e}")
        return None

def convert_midi_to_wav(midi_file, wav_file):
    """Conversion MIDI -> WAV via FluidSynth"""
    soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2" # Chemin Colab/Linux standard
    if not os.path.exists(soundfont):
        soundfont = "default.sf2" # Essai local
    
    cmd = ["fluidsynth", "-ni", soundfont, midi_file, "-F", wav_file, "-r", "44100"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

# ==============================================================================
# 5. INTERFACE PRINCIPALE
# ==============================================================================

st.title("🎼 APPASSIONATO")
st.markdown("### Génération de Musique Polyphonique avec MuseGAN")
st.write("Ce modèle génère une partition originale pour 5 instruments. Choisissez votre style dans le menu à gauche et lancez la création.")

# Chargement
generator = load_model()
if generator is None:
    st.error(f"⚠️ ERREUR : Le fichier `{CHECKPOINT_FILE}` est manquant. Veuillez le téléverser.")
    st.stop()

# Bouton d'action
col_act1, col_act2 = st.columns([1, 4])
with col_act1:
    gen_btn = st.button("🎹 GÉNÉRER", type="primary", use_container_width=True)

if gen_btn:
    progress_text = "L'IA compose... (Calcul des tenseurs 3D)"
    my_bar = st.progress(0, text=progress_text)

    # 1. Génération
    z = torch.randn(nb_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_data = generator(z).cpu().numpy()
        # generated_data shape: (nb_samples, 5, 64, 72)
    
    my_bar.progress(30, text="Assemblage des pistes...")

    # 2. Reshape correct pour coller les blocs de temps
    # On passe de (N, Tracks, Time, Pitch) -> (Tracks, N * Time, Pitch)
    # Transpose: (1, 0, 2, 3) -> (Tracks, N, Time, Pitch)
    # Reshape: (Tracks, TotalTime, Pitch)
    full_song = generated_data.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
    
    # 3. Algorithme Seuil Dynamique
    max_conf = np.max(full_song)
    threshold = 0.5
    if max_conf < 0.5:
        threshold = max_conf * 0.9
        
    binary_song = full_song > threshold
    
    my_bar.progress(60, text=f"Application des instruments ({style_choice})...")

    # 4. Création MIDI (CORRIGÉE : StandardTrack + Vélocité)
    tracks = []
    for idx, (name, program, is_drum) in enumerate(zip(track_names, programs, is_drums)):
        # Padding
        padded = np.pad(binary_song[idx], ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches)))
        
        # Upscaling temporel
        upscaled = np.repeat(padded, UPSCALE_FACTOR, axis=0) 
        
        # Conversion bool -> int (0 ou 100) pour éviter l'erreur 0..127
        pianoroll_standard = upscaled.astype(np.uint8) * 100
        
        tracks.append(pypianoroll.StandardTrack(
            name=name, program=program, is_drum=is_drum, pianoroll=pianoroll_standard
        ))
    
    # Création Multitrack (CORRIGÉE : Tempo (T, 1))
    # On force la dimension (TotalTime, 1) pour le tempo
    tempo_array = np.full((upscaled.shape[0], 1), tempo)

    multitrack = pypianoroll.Multitrack(
        tracks=tracks,
        tempo=tempo_array,
        resolution=EXPORT_RESOLUTION
    )
    
    midi_filename = "output.mid"
    wav_filename = "output.wav"
    multitrack.write(midi_filename)
    
    # 5. Conversion Audio
    my_bar.progress(80, text="Conversion Audio (FluidSynth)...")
    has_audio = convert_midi_to_wav(midi_filename, wav_filename)
    my_bar.progress(100, text="Terminé !")
    my_bar.empty()

    # --- RÉSULTATS ---
    st.divider()
    
    # Colonne Gauche : Lecteur et Infos
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.success("✨ Composition terminée")
        st.metric("Confiance IA", f"{max_conf:.2%}")
        
        st.markdown("#### 🎧 Écouter")
        if has_audio and os.path.exists(wav_filename):
            st.audio(wav_filename, format='audio/wav')
        else:
            st.warning("Audio web non disponible (Fluidsynth manquant).")
            
        with open(midi_filename, "rb") as f:
            st.download_button("⬇️ Télécharger le MIDI", f, file_name="appassionato.mid", use_container_width=True)

    # Colonne Droite : Visualisation
    with c2:
        st.markdown("#### 📊 Visualisation (Piano-roll)")
        # On affiche la piste principale (souvent Piano ou Guitare, index 0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(binary_song[0].T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Partition : {track_names[0]}")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Notes (Pitch)")
        st.pyplot(fig)

    # Affichage détaillé des pistes
    with st.expander("Voir les détails des 5 pistes"):
        for i in range(n_tracks):
            st.write(f"**Piste {i+1} : {track_names[i]}** (Programme MIDI : {programs[i]})")