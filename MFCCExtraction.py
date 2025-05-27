import os
import librosa
import numpy as np
from collections import defaultdict

# Definicije skupina fonema
voiced = {'a', 'e', 'i', 'o', 'u', 'b', 'd', 'dž', 'đ', 'g', 'j', 'l', 'lj', 'm',
          'n', 'nj', 'r', 'v', 'z', 'ž', 'a:', 'e:', 'i:', 'o:', 'u:'}
unvoiced = {'c', 'ć', 'č', 'f', 'h', 'k', 'p', 's', 'š', 't'}
silence = {'sil'}

# Folderi s podacima
lab_folder = "VEPRAD database"
wav_folder = os.path.join(lab_folder, "wav")

# Parametri MFCC
n_fft = 512
hop_length = 256

# Korak 1: Učitaj sve .wav datoteke i zapamti signale i sample rate
# Ovaj dio je neovisan o poslije skupljanju segmenata
audio_data = {}  # filename (bez .wav) -> (signal, sr)

for file in os.listdir(wav_folder):
    if file.endswith(".wav"):
        wav_path = os.path.join(wav_folder, file)
        file_key = file.replace(".wav", ".lab")
        try:
            # cijela audio datoteka kao niz uzoraka (y) i daje ti sample rate (sr) — to je fiksni podatak za tu datoteku.
            y, sr = librosa.load(wav_path, sr=None)  # koristi originalni sample rate
            audio_data[file_key] = (y, sr)
        except Exception as e:
            print(f"Upozorenje: Ne mogu učitati {file}, preskačem. Detalji: {e}")

print(f"Učitano {len(audio_data)} .wav datoteka.")

# Korak 2: Parsiraj .lab datoteke i grupiraj intervale
intervals_by_file = defaultdict(list)

# Učitavanje .lab datoteke ako gore postoji istoimena .wav datoteka
# Osim toga daje se label ovisno o gore definiranim klasama (mozda treba dodat unutra jos neke stvari vidio sam da u .lab
# ponekad ima cc ili uzdah pa nez sta s time trenutno)
for file in os.listdir(lab_folder):
    if file.endswith(".lab") and file in audio_data:
        lab_path = os.path.join(lab_folder, file)
        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                start_us, end_us, label = line.strip().split()
                start = float(start_us) / 1_000_000
                end = float(end_us) / 1_000_000

                if label in voiced:
                    label_tag = "voiced"
                elif label in unvoiced:
                    label_tag = "unvoiced"
                else:
                    label_tag = "silence"

                intervals_by_file[file].append((start, end, label_tag))

print(f"Obrađeno {len(intervals_by_file)} .lab datoteka.")

# Korak 3: Izračunaj MFCC za svaki interval
mfcc_features = []
labels = []

for file, segments in intervals_by_file.items():
    y, sr = audio_data[file]
    for start, end, label_tag in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < n_fft:
            continue  # preskoči samo one koji su kraći od jednog prozora

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfcc_mean = np.mean(mfcc, axis=1)

        mfcc_features.append(mfcc_mean)
        labels.append(label_tag)

# Konverzija u numpy polja
mfcc_features = np.array(mfcc_features)
labels = np.array(labels)

# Ispis rezultata
print("Oblik MFCC značajki:", mfcc_features.shape)
print("Prvih 10 oznaka:", labels[:10])
print("Prvih nekoliko MFCC vrijednosti:", mfcc_features[:10])

# Sada u mffc_features arrayu imam MFCC značajke za pojedini segment a u labels oznake za te segmenete
# na istom indexu u oba arrayu su značajke i oznaka

