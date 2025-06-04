import os
import numpy as np
import librosa

# 1) Definicija fonema
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

# 2) Čitanje .lab datoteke
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

    for start_s, end_s, ph_label in segments:
        start_sample = int(start_s * sr)
        end_sample   = int(end_s * sr)

        if end_sample <= start_sample:
            continue

        segment = y[start_sample:end_sample]
        segment_len = len(segment)

        # Segment mora imati barem 32 uzorka
        if segment_len < 32:
            continue

        # Najveća snaga broja 2 koja stane u segment
        max_power = int(np.floor(np.log2(segment_len)))
        n_fft = 2**max_power

        # Hop length postavi na polovicu FFT duljine (50% overlap)
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

        # Ako je rezultat premalen, preskoči (npr. neka numerička greška)
        if mfcc.shape[1] < 1:
            continue

        # Prosjek po vremenskoj osi → vektor duljine 13
        mfcc_mean = np.mean(mfcc, axis=1)

        if ph_label in voiced_set:
            labels.append(1)
            features.append(mfcc_mean)
        elif ph_label in unvoiced_set:
            labels.append(0)
            features.append(mfcc_mean)
        else:
            continue  # nepoznati fonemi se preskaču

    assert len(features) == len(labels), "Broj značajki i labela nije jednak!"

    return np.array(features), np.array(labels)


# 4) Glavni dio: prolaz kroz sve datoteke
data_dir = 'MFCC-and-Bayes-klasification-of-sound/VEPRAD database'
X_all = []
y_all = []

for fname in os.listdir(data_dir):
    if fname.endswith('.wav'):
        wav_fp = os.path.join(data_dir, fname)
        lab_fp = wav_fp.replace('.wav', '.lab')
        if not os.path.exists(lab_fp):
            continue
        mfcc_feats, frame_labels = extract_features_and_labels_from_segments(wav_fp, lab_fp)
        print(len(mfcc_feats), "MFCC features extracted from", fname)
        print(len(frame_labels), "labels extracted from", fname)
        if mfcc_feats.size == 0:
            continue
        X_all.append(mfcc_feats)
        y_all.append(frame_labels)

# 5) Spajanje i spremanje
if X_all:
    X = np.vstack(X_all)
    y = np.hstack(y_all)

    print("Ukupno segmenata:", X.shape[0], "   Dimenzija značajki:", X.shape[1])
    print("Broj labela:", y.shape[0])
    assert X.shape[0] == y.shape[0], "ZNAČAJKE I Labele nisu istog broja!"

    output_file = 'features_labels_segment.npz'
    np.savez_compressed(output_file, X=X, y=y)
    print(f"Spremljeni podaci u {output_file}")
else:
    print("Nema dovoljno valjanih segmenata za obradu.")
