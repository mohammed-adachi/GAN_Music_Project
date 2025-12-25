# ==============================================================================
# APPLICATION STREAMLIT - GÉNÉRATEUR DE MUSIQUE MUSEGAN
# ==============================================================================
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pypianoroll
from io import BytesIO
import subprocess
import tempfile

# ==============================================================================
# CONFIGURATION PAGE
# ==============================================================================
st.set_page_config(
    page_title="🎵 MuseGAN - Générateur de Musique",
    page_icon="🎼",
    layout="wide"
)

# ==============================================================================
# PARAMÈTRES GLOBAUX
# ==============================================================================
n_tracks = 5
n_pitches = 72
lowest_pitch = 24
n_measures = 4
beat_resolution = 4
measure_resolution = 16

track_names = ["Piano 1", "Piano 2", "Bass", "Strings/Pad", "Soft Drums"]
programs = [0, 0, 32, 49, 0]
is_drums = [False, False, False, False, True]

tempo = 65
EXPORT_RESOLUTION = 24
UPSCALE_FACTOR = EXPORT_RESOLUTION // beat_resolution

latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "musegan_checkpoint_4.pth"

# ==============================================================================
# ARCHITECTURE DU MODÈLE
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
        
        self.transconv0 = GeneraterBlock(self.z_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        
        self.transconv4 = nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
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
# CACHE - CHARGER LE MODÈLE UNE SEULE FOIS
# ==============================================================================
@st.cache_resource
def load_generator():
    if not os.path.exists(CHECKPOINT_FILE):
        st.error(f"❌ Le fichier '{CHECKPOINT_FILE}' est introuvable!")
        st.stop()
    
    generator = Generator().to(device)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

# ==============================================================================
# FONCTION POUR GÉNÉRER LE BRUIT ALÉATOIRE (AVANT MODÈLE)
# ==============================================================================
def generate_random_noise(nb_generations=2):
    """Génère du bruit aléatoire (vecteur latent) sans passer par le modèle"""
    # Générer du bruit aléatoire avec la même forme qu'une sortie du générateur
    random_segments = []
    
    for i in range(nb_generations):
        # Créer du bruit aléatoire avec la même forme que le générateur
        # Shape: (1, n_tracks, n_measures * measure_resolution, n_pitches)
        noise = np.random.uniform(0, 1, (1, n_tracks, n_measures * measure_resolution, n_pitches))
        random_segments.append(noise)
    
    full_noise = np.concatenate(random_segments, axis=2)
    
    # Seuil bas pour montrer plus de bruit
    threshold_noise = 0.5
    sample_binary_noise = full_noise[0] > threshold_noise
    total_notes_noise = np.sum(sample_binary_noise)
    
    return full_noise, sample_binary_noise, total_notes_noise, threshold_noise

# ==============================================================================
# FONCTION DE GÉNÉRATION AVEC MODÈLE
# ==============================================================================
def generate_music(nb_generations=2):
    generator = load_generator()
    generated_segments = []
    
    with torch.no_grad():
        for i in range(nb_generations):
            latent_vector = torch.randn(1, latent_dim).to(device)
            output = generator(latent_vector)
            generated_segments.append(output.cpu().numpy())
    
    full_song = np.concatenate(generated_segments, axis=2)
    
    # Analyse de la confiance
    max_prob = np.max(full_song)
    mean_prob = np.mean(full_song)
    
    # Seuil dynamique
    base_threshold = 0.55
    if max_prob < base_threshold:
        threshold = max_prob * 0.9
    else:
        threshold = base_threshold
    
    sample_binary = full_song[0] > threshold
    total_notes = np.sum(sample_binary)
    
    return full_song, sample_binary, total_notes, max_prob, threshold

# ==============================================================================
# FONCTION POUR CRÉER MULTITRACK
# ==============================================================================
def create_multitrack_midi(sample_binary, output_filename="generated_music.mid"):
    tracks_list = []
    
    for i in range(n_tracks):
        pianoroll_padded = np.pad(
            sample_binary[i], ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
        )
        pianoroll_upscaled = np.repeat(pianoroll_padded, UPSCALE_FACTOR, axis=0)
        
        track = pypianoroll.BinaryTrack(
            name=track_names[i],
            program=programs[i],
            is_drum=is_drums[i],
            pianoroll=pianoroll_upscaled
        )
        tracks_list.append(track)
    
    total_length_upscaled = sample_binary.shape[1] * UPSCALE_FACTOR
    
    multitrack = pypianoroll.Multitrack(
        tracks=tracks_list,
        tempo=np.full(total_length_upscaled, tempo),
        resolution=EXPORT_RESOLUTION
    )
    
    multitrack.write(output_filename)
    return multitrack

# ==============================================================================
# FONCTION POUR CONVERTIR MIDI EN WAV (SANS INSTALLATION)
# ==============================================================================
@st.cache_data
def midi_to_wav_with_soundfont(midi_file, output_wav, soundfont_path=None):
    """
    Convertit MIDI en WAV en utilisant fluidsynth si disponible.
    Sinon, crée un fichier audio simple.
    """
    try:
        # Vérifier si fluidsynth est disponible
        result = subprocess.run(['which', 'fluidsynth'], capture_output=True)
        if result.returncode == 0:
            # Trouver une soundfont par défaut
            soundfont_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/share/sounds/sf2/FluidR3_GS.sf2',
                '/usr/share/sounds/soundfonts/FluidR3_GM.sf2',
            ]
            
            soundfont = None
            for sf in soundfont_paths:
                if os.path.exists(sf):
                    soundfont = sf
                    break
            
            if soundfont:
                cmd = f'fluidsynth -ni "{soundfont}" "{midi_file}" -F "{output_wav}" -r 44100'
                subprocess.run(cmd, shell=True, capture_output=True)
                if os.path.exists(output_wav):
                    return True
    except:
        pass
    
    return False

# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
st.title("🎵 Générateur de Musique MuseGAN")
st.write("Générez de la musique calme avec un modèle d'IA entraîné!")

# Sidebar - Paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    nb_generations = st.slider("Nombre de blocs à générer", 1, 5, 2)
    st.write(f"**Durée estimée**: ~{nb_generations * 15} secondes")

# Bouton de génération
col1, col2, col3 = st.columns(3)
with col1:
    show_noise_btn = st.button("🎲 Voir le bruit aléatoire (AVANT modèle)", use_container_width=True)
with col2:
    generate_btn = st.button("🎹 Générer Musique (AVEC modèle)", use_container_width=True)

# === AFFICHER LE BRUIT ALÉATOIRE AVANT LE MODÈLE ===
if show_noise_btn:
    st.subheader("🎲 Étape 1: Bruit Aléatoire (Avant le Modèle)")
    st.info("Ceci est le bruit aléatoire AVANT qu'il ne passe par le modèle. C'est bruyant et sans structure!")
    
    with st.spinner("⏳ Génération du bruit aléatoire..."):
        full_noise, sample_binary_noise, total_notes_noise, threshold_noise = generate_random_noise(2)
    
    # Créer le fichier MIDI pour le bruit
    with st.spinner("💾 Création du fichier MIDI (bruit)..."):
        multitrack_noise = create_multitrack_midi(sample_binary_noise, "noise_before_model.mid")
    
    # Convertir en WAV
    wav_noise = "noise_before_model.wav"
    with st.spinner("🔄 Conversion en audio..."):
        audio_noise_available = midi_to_wav_with_soundfont("noise_before_model.mid", wav_noise)
    
    # Lecteur audio - BRUIT
    col1, col2 = st.columns(2)
    with col1:
        st.write("**🔊 Écoutez le bruit aléatoire:**")
        if audio_noise_available and os.path.exists(wav_noise):
            with open(wav_noise, "rb") as f:
                audio_data = f.read()
            st.audio(audio_data, format="audio/wav")
        else:
            st.warning("Impossible de générer l'audio du bruit")
    
    with col2:
        st.metric("Notes totales (bruit)", int(total_notes_noise))
    
    # Visualisation bruit
    st.subheader("📊 Piano Roll du Bruit Aléatoire")
    fig_noise, axes_noise = plt.subplots(n_tracks, 1, figsize=(14, 8))
    
    for i in range(n_tracks):
        axes_noise[i].imshow(sample_binary_noise[i].T, aspect='auto', origin='lower', cmap='Reds')
        axes_noise[i].set_ylabel(track_names[i], fontsize=10, fontweight='bold')
    
    plt.suptitle("Piano Roll - BRUIT ALÉATOIRE (Avant le Modèle)", fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()
    st.pyplot(fig_noise)

# === GÉNÉRER LA MUSIQUE AVEC LE MODÈLE ===
if generate_btn:
    st.subheader("🎹 Étape 2: Musique Générée (Après le Modèle)")
    st.success("Le modèle a transformé le bruit aléatoire en MUSIQUE structurée!")
    
    with st.spinner("⏳ Génération en cours..."):
        full_song, sample_binary, total_notes, max_prob, threshold = generate_music(2)
    
    # Afficher les statistiques
    st.success("✅ Musique générée avec succès!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confiance Max", f"{max_prob:.4f}")
    with col2:
        st.metric("Seuil Utilisé", f"{threshold:.4f}")
    with col3:
        st.metric("Notes Totales", int(total_notes))
    with col4:
        st.metric("Pistes", n_tracks)
    
    # Créer et sauvegarder le MIDI
    with st.spinner("💾 Création du fichier MIDI..."):
        multitrack = create_multitrack_midi(sample_binary, "generated_music.mid")
    
    # Tentez de convertir en WAV pour le lecteur audio
    wav_file = "generated_music.wav"
    audio_available = False
    
    with st.spinner("🔄 Conversion en audio (modèle)..."):
        audio_available = midi_to_wav_with_soundfont("generated_music.mid", wav_file)
    
    # Lecteur audio intégré - MUSIQUE
    st.subheader("🔊 Écoutez la Musique Générée")
    if audio_available and os.path.exists(wav_file):
        with open(wav_file, "rb") as f:
            audio_data = f.read()
        st.audio(audio_data, format="audio/wav")
    else:
        st.warning("💡 Impossible de générer l'audio. Assurez-vous que fluidsynth est installé.")
    
    # Visualisation
    st.subheader("📊 Piano Roll de la Musique Générée")
    fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 10))
    
    for i in range(n_tracks):
        axes[i].imshow(sample_binary[i].T, aspect='auto', origin='lower', cmap='Greens')
        axes[i].set_ylabel(track_names[i], fontsize=10, fontweight='bold')
        axes[i].set_xlabel("Temps")
    
    axes[-1].set_xlabel("Temps (ticks)")
    plt.suptitle("Piano Roll - MUSIQUE GÉNÉRÉE (Après le Modèle)", fontsize=14, fontweight='bold', color='green')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Télécharger le fichier MIDI
    with open("generated_music.mid", "rb") as f:
        midi_bytes = f.read()
    
    st.download_button(
        label="⬇️ Télécharger fichier MIDI",
        data=midi_bytes,
        file_name="generated_music.mid",
        mime="audio/midi",
        use_container_width=True
    )
    
    # Afficher les détails des pistes
    st.subheader("🎼 Détails des pistes")
    for i, track_name in enumerate(track_names):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{track_name}**")
        with col2:
            notes_count = np.sum(sample_binary[i])
            st.write(f"Notes: {int(notes_count)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎶 Créé avec MuseGAN | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
