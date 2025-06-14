import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

voiced_set = {
    'a','e','i','o','u', 'a:','e:','i:','o:','u:',
    'm','n','nj','lj','r','l','j',
    'b','d','g','dz','dZ', 'dZ\'',
    'z','Z','v'
}
unvoiced_set = {
    'p','t','k','c','tS','tS\'','cc',
    'f','s','S','x',
    'sil','silh','uzdah'
}

def read_lab(lab_path):
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start_unit = int(parts[0])
            end_unit   = int(parts[1])
            ph_label   = parts[2]

            start_s = start_unit * 1e-7
            end_s   = end_unit * 1e-7
            segments.append((start_s, end_s, ph_label))
    return segments

def extract_features_and_labels_from_segments(wav_path, lab_path):
    y, sr = librosa.load(wav_path, sr=16000)
    segments = read_lab(lab_path)

    n_mfcc = 13
    features = []
    labels = []
    phonemes = []

    for start_s, end_s, ph_label in segments:
        start_sample = int(start_s * sr)
        end_sample   = int(end_s * sr)

        if end_sample <= start_sample:
            continue

        segment = y[start_sample:end_sample]
        segment_len = len(segment)

        if segment_len < 32:
            continue

        max_power = int(np.floor(np.log2(segment_len)))
        n_fft = 2**max_power
        hop_length = max(1, n_fft // 2)

        try:
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
                htk=True
            )
        except Exception as e:
            print(f"MFCC error on segment {ph_label} ({start_s:.4f}-{end_s:.4f}): {e}")
            continue

        if mfcc.shape[1] < 1:
            continue

        mfcc_mean = np.mean(mfcc, axis=1)

        if ph_label in voiced_set:
            labels.append(1)
            features.append(mfcc_mean)
            phonemes.append(ph_label)
        elif ph_label in unvoiced_set:
            labels.append(0)
            features.append(mfcc_mean)
            phonemes.append(ph_label)
        else:
            continue

    assert len(features) == len(labels) == len(phonemes), "Broj značajki, labela i fonema nije jednak!"

    return np.array(features), np.array(labels), np.array(phonemes)


# 1. Učitaj model
clf = joblib.load('naive_bayes_model.pkl')

# 2. Učitaj značajke, labele i foneme
mfcc_feats, frame_labels, phoneme_labels = extract_features_and_labels_from_segments(
    r"MFCC-and-Bayes-klasification-of-sound\VEPRAD database\sm04010105109.wav",
    r"MFCC-and-Bayes-klasification-of-sound\VEPRAD database\sm04010105109.lab"
)

# 3. Predikcija
predicted_labels = clf.predict(mfcc_feats)

# 4. Ispis usporedbe prvih 30
print("=== Predikcije po segmentima (0 = unvoiced, 1 = voiced) ===")
for i in range(min(30, len(predicted_labels))):
    print(f"{i+1:02d}. Fonem: {phoneme_labels[i]:8} | Predikcija: {predicted_labels[i]} | Stvarna: {frame_labels[i]}")

# 5. Grafički prikaz „lente” prvih 30 stvarnih i predikcija
max_segments = min(30, len(predicted_labels))
data = np.vstack([frame_labels[:max_segments], predicted_labels[:max_segments]])
cmap = ListedColormap(['orange', 'green'])  # 0→narančasta, 1→zelena

fig, ax = plt.subplots(figsize=(15, 2))
im = ax.imshow(
    data,
    aspect='auto',
    cmap=cmap,
    interpolation='nearest'
)

# crne linije (ili bijele, ovisno o pozadini) između stupaca
for x in range(max_segments + 1):
    ax.axvline(x - 0.5, color='white', linewidth=0.8)

# linija koja razdvaja stvarne i predikcije
ax.axhline(0.5, color='black', linewidth=1)

ax.set_yticks([0, 1])
ax.set_yticklabels(['Stvarna', 'Predikcija'])
ax.set_xlabel('Segment index')
ax.set_title('Prvih 30 segmenata – zelena = voiced (1), narančasta = unvoiced (0)')
ax.set_xticks(np.arange(max_segments))
ax.set_xticklabels(np.arange(1, max_segments+1), rotation=90, fontsize=5)
ax.set_ylim(-0.5, 1.5)
plt.tight_layout()
plt.show()
