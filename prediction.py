import librosa
import numpy as np
import joblib

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

# 3) Ekstrakcija MFCC po segmentima
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
clf = joblib.load('naive_bayes_model1.pkl')

# 2. Učitaj značajke, labele i foneme
mfcc_feats, frame_labels, phoneme_labels = extract_features_and_labels_from_segments(
    r"C:\Users\Patrik\Desktop\Projekt ipsic\MFCC-and-Bayes-klasification-of-sound\VEPRAD database\wav\sm04010103201.wav",
    r"C:\Users\Patrik\Desktop\Projekt ipsic\MFCC-and-Bayes-klasification-of-sound\VEPRAD database\wav\sm04010103201.lab"
)

# 3. Predikcija
predicted_labels = clf.predict(mfcc_feats)

# 4. Ispis usporedbe
print("=== Predikcije po segmentima (0 = unvoiced, 1 = voiced) ===")
for i in range(min(20, len(predicted_labels))):
    print(f"{i+1:02d}. Fonem: {phoneme_labels[i]:8} | Predikcija: {predicted_labels[i]} | Stvarna: {frame_labels[i]}")
