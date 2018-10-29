import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.lib.display import Audio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def mfcc_dist(a,b):
      dist = 0
      for x, y in zip(a,b):
          dist = dist + (x - y) * (x - y)
      return dist**0.5

mcc_list = []
# load
for filename in [u"cuts/bosnian3/Wednesday.mp3", u"cuts/english368/Wednesday.mp3"]:
	y, sr = librosa.load(filename)

	# calculate mfcc
	Y = librosa.stft(y)
	mfccs = librosa.feature.mfcc(y,
								sr=16000,
								n_mfcc=13)

	mcc_list.append((mfccs,y,sr))

mcc1 = mcc_list[0][0].T
mcc2 = mcc_list[1][0].T

max_size = max(mcc1.shape[0],mcc2.shape[0])

# mcc1 = librosa.util.pad_center(mcc1, max_size, axis=0)
# mcc2 = librosa.util.pad_center(mcc2, max_size, axis=0)

print(mcc1.shape)
print(mcc2.shape)

distance, path = fastdtw(mcc1, mcc2, dist=mfcc_dist)

print(path)

# # Build reconstruction mappings,
# n_mfcc = mfccs.shape[0]
# n_mel = 128
# dctm = librosa.filters.dct(n_mfcc, n_mel)
# n_fft = 2048
# mel_basis = librosa.filters.mel(sr, n_fft)

# # Empirical scaling of channels to get ~flat amplitude mapping.
# bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

# # Reconstruct the approximate STFT squared-magnitude from the MFCCs.
# recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

# # Impose reconstructed magnitude on white noise STFT.
# excitation = np.random.randn(y.shape[0])
# E = librosa.stft(excitation)
# recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

# # Output
# librosa.output.write_wav('output.wav', recon, sr)

# plt.style.use('seaborn-darkgrid')
# plt.figure(1)
# plt.subplot(211)
# librosa.display.waveplot(y, sr)
# plt.subplot(212)
# librosa.display.waveplot(recon,sr)
# plt.show()