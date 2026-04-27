#2) Pamulapati Mahesh Babu (35045493)

# -----------------------------
# 7. Class distribution
# -----------------------------
plt.figure(figsize=(10, 4))
sns.countplot(x="emotion", data=df, order=sorted(df["emotion"].unique()))
plt.title("Emotion Class Distribution")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# 8. Audio augmentation functions
# -----------------------------
def add_noise(signal, noise_factor=0.005):
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def shift_signal(signal, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(signal))
    return np.roll(signal, shift)

def change_pitch(signal, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

# -----------------------------
# 9. Feature extraction settings
# -----------------------------
SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
MAX_PAD_LEN = 130   # fixed time frames for MFCC matrix

def load_audio_fixed(path, sr=SAMPLE_RATE, duration=DURATION):
    signal, sr = librosa.load(path, sr=sr)

    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, padding), mode='constant')

    return signal, sr

def extract_mfcc(signal, sr, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T   # shape: (time, features)

