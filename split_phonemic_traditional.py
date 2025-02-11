#%%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import IPython.display as ipd
#%%
# Load audio file
audio_path = "libri-audio/8842-304647-0013.wav"  # Replace with your audio file
y, sr = librosa.load(audio_path, sr=None)

# Compute the Short-Time Fourier Transform (STFT)
stft = np.abs(librosa.stft(y, n_fft=512, hop_length=128))

# Compute Spectral Flux (change in frequency over time)
sf = np.sum(np.diff(stft, axis=1) ** 2, axis=0)

# Normalize spectral flux
sf = (sf - np.min(sf)) / (np.max(sf) - np.min(sf))

# Find peaks in spectral flux (corresponding to phoneme transitions)
peaks, _ = find_peaks(sf, height=0.1, distance=5)  # Tune height and distance

# Convert peak indices to time
peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=128)

# Plot the waveform and detected phoneme boundaries
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(peak_times, ymin=-1, ymax=1, color='r', linestyle='dashed', label="Phoneme Boundaries")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Detected Phoneme Segments")
plt.legend()
plt.show()

# Print detected phoneme timestamps
print("Phoneme Boundaries (seconds):", peak_times)

# %%
# split audio based on phoneme boundaries
audio_segments = np.split(y, [int(sr * t) for t in peak_times])
# Pad segments with 0.1 seconds of silence on both sides
# audio_segments = [np.concatenate([np.zeros(int(0.1 * sr)), segment, np.zeros(int(0.1 * sr))]) for segment in audio_segments]
for i, segment in enumerate(audio_segments):
    print(f"Segment {i}: {len(segment)} samples")
    ipd.display(ipd.Audio(segment, rate=sr))
# %%
# plot audio_segments as one audio
audio = np.concatenate(audio_segments)
ipd.display(ipd.Audio(audio, rate=sr))
# %%
# plot audio_segments, make each segment final to be longer
longer_segments = [np.concatenate([segment, np.tile(segment[-int(0.1 * sr):],10)]) for segment in audio_segments]
for i, segment in enumerate(longer_segments):
    print(f"Segment {i}: {len(segment)} samples")
    ipd.display(ipd.Audio(segment, rate=sr))
# %%
