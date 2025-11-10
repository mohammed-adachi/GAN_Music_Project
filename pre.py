import os
from pathlib import Path
import numpy as np
import pypianoroll
import pretty_midi
import warnings

# --- CONFIGURATION ---
INPUT_DIR = "lmd_full/"
# On change le nom pour sauvegarder plusieurs parties
OUTPUT_PREFIX = "lmdfull/musegan_training_data_part"
RECURSIVE = True
START_FILE = 120000
END_FILE = None   # jusqu'à la fin
PLATFORM_NAME = "kaggle"
# --- PARAMÈTRES ---
FAMILY_NAMES = ["drum", "bass", "guitar", "string", "piano"]
FAMILY_THRESHOLDS = [(1, 12), (1, 48), (1, 78), (1, 78), (1, 78)]
RESOLUTION = 24

# --- NOUVEAU PARAMÈTRE POUR LA SAUVEGARDE ---
# On sauvegarde tous les 20 000 segments trouvés pour ne pas remplir la RAM
SAVE_BATCH_SIZE = 20000

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Les fonctions check_which_family et segment_quality restent les mêmes
def check_which_family(instrument):
    program = instrument.program; is_drum = instrument.is_drum
    def is_piano(): return not is_drum and ((0 <= program <= 7) or (16 <= program <= 23))
    def is_guitar(): return 24 <= program <= 31
    def is_bass(): return 32 <= program <= 39
    def is_string(): return 40 <= program <= 51
    return np.array([is_drum, is_bass(), is_guitar(), is_string(), is_piano()])

def segment_quality(pianoroll_segment, threshold_pitch, threshold_beats):
    pitch_sum = np.sum(np.sum(pianoroll_segment, axis=0) > 0)
    beat_sum = np.sum(np.sum(pianoroll_segment, axis=1) > 0)
    return ((pitch_sum >= threshold_pitch) and (beat_sum >= threshold_beats)), (pitch_sum, beat_sum)

def save_batch(segments, part_number):
    """Fonction pour sauvegarder un lot de segments."""
    if not segments:
        return
    output_filename = f"{OUTPUT_PREFIX}_{part_number}.npz"
    print("-" * 30)
    print(f"Sauvegarde du lot n°{part_number} avec {len(segments)} segments...")
    print(f"Fichier de sortie : {output_filename}")
    np.savez_compressed(output_filename, *segments)
    print("Lot sauvegardé avec succès !")
    print("-" * 30)

def main():
    print("Début du prétraitement ROBUSTE avec sauvegarde par lots...")
    print(f"Traitement des fichiers de l'index {START_FILE} à {END_FILE or 'la fin'}.")
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERREUR : Le dossier d'entrée n'existe pas : {input_path}")
        return

    print("Scan de tous les fichiers MIDI (cela peut prendre un moment)...")
    all_filenames = sorted(list(input_path.rglob("*.mid"))) if RECURSIVE else sorted(list(input_path.glob("*.mid")))
    print(f"{len(all_filenames)} fichiers MIDI trouvés au total.")

    # On utilise une liste temporaire pour un lot
    files_to_process = all_filenames[START_FILE:END_FILE]
    batch_of_segments = []
    part_number = 1
    total_files_in_chunk = len(files_to_process)
    print(f"Ce script va traiter {total_files_in_chunk} fichiers.")

    for i, filename in enumerate(files_to_process):
        if (i + 1) % 500 == 0:
            print(f"Progression ({PLATFORM_NAME}): {i + 1} / {total_files_in_chunk} fichiers traités...")

        

                # ... (Le reste de la logique de traitement de segment est identique) ...
        try:
            pm = pretty_midi.PrettyMIDI(str(filename))
            downbeats = pm.get_downbeats()
            if len(downbeats) < 5: continue
            
            multitrack = pypianoroll.from_pretty_midi(pm, resolution=RESOLUTION)
            num_bars = len(downbeats) - 1
            
            for bar_idx in range(0, num_bars - 4, 2):
                start_tick = int(downbeats[bar_idx] * RESOLUTION)
                end_tick = int(downbeats[bar_idx + 4] * RESOLUTION)
                
                segment_tracks = {name: None for name in FAMILY_NAMES}
                best_scores = {name: -1 for name in FAMILY_NAMES}
                
                for track in multitrack.tracks:
                    family_map = check_which_family(track)
                    in_family_idx = np.where(family_map)[0]
                    if not len(in_family_idx): continue
                    
                    family_name = FAMILY_NAMES[in_family_idx[0]]
                    family_thresholds = FAMILY_THRESHOLDS[in_family_idx[0]]
                    
                    pianoroll_segment = track.pianoroll[start_tick:end_tick]
                    is_ok, score = segment_quality(pianoroll_segment, family_thresholds[0], family_thresholds[1])
                    
                    if is_ok and sum(score) > best_scores[family_name]:
                        best_scores[family_name] = sum(score)
                        segment_tracks[family_name] = pianoroll_segment
                
                valid_instruments_count = sum(1 for track in segment_tracks.values() if track is not None)
                if valid_instruments_count >= 2:
                    final_segment = []
                    TARGET_LENGTH = 4 * RESOLUTION
                    
                    for name in FAMILY_NAMES:
                        if segment_tracks[name] is not None:
                            pianoroll = segment_tracks[name]
                            padded_pianoroll = np.zeros((TARGET_LENGTH, 128), dtype=bool)
                            actual_length = min(TARGET_LENGTH, pianoroll.shape[0])
                            padded_pianoroll[:actual_length, :] = pianoroll[:actual_length, :] > 0
                            final_segment.append(padded_pianoroll[:, :, np.newaxis])
                        else:
                            final_segment.append(np.zeros((TARGET_LENGTH, 128, 1), dtype=bool))
                    
                    compiled_segment = np.concatenate(final_segment, axis=2)
                    batch_of_segments.append(compiled_segment)
                    
                    if len(batch_of_segments) >= SAVE_BATCH_SIZE:
                        save_batch(batch_of_segments, part_number)
                        part_number += 1
                        batch_of_segments.clear()
        except Exception:
            continue
    
    print("Fin du traitement des fichiers. Sauvegarde du dernier lot...")
    save_batch(batch_of_segments, part_number)
    print(f"Prétraitement terminé pour {PLATFORM_NAME} !")

if __name__ == "__main__":
    main()