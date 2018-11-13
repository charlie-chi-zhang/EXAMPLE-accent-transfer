import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.lib.display import Audio
from fastdtw import fastdtw
from mutagen.mp3 import MP3
from scipy.spatial.distance import euclidean
import wave
from pydub import AudioSegment
import subprocess

def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def mfcc_dist(a,b):
      dist = 0
      for x, y in zip(a,b):
          dist = dist + (x - y) * (x - y)
      return dist**0.5

def get_duration_wav(wav_filename):
    f = wave.open(wav_filename, 'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    return duration

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def gen_mfcc(filename, offset = 0, duration = 0.005):

	sr = 20000
	n_mfcc=25

	y, sr = librosa.load(filename, 
						sr = sr, 
						offset = offset, 
						duration = duration
						)

	# calculate mfcc
	#Y = librosa.stft(y)
	mfccs = librosa.feature.mfcc(y,
								sr=sr,
								n_mfcc=n_mfcc)
	# print(mfccs, y.shape, sr)
	return mfccs, y, sr

def allign_sample_target(sample_mfcc, target_mfcc, max_size = None):

	max_size = max(sample_mfcc.T.shape[0],target_mfcc.T.shape[0])
	min_size = min(sample_mfcc.T.shape[0],target_mfcc.T.shape[0])

	# print(max_size-min_size)
	# print(target_mfcc.T.shape)
	# print(sample_mfcc.T.shape)

	sample = np.pad(sample_mfcc.T, (max_size-min_size,0), mode = "constant")
	target = np.pad(target_mfcc.T, (max_size-min_size,0), mode = "constant")

	# print(sample.shape)

	distance, path = fastdtw(sample, target, dist=mfcc_dist)

	sample_aligned = [sample[i] for (i, j) in path]
	target_aligned = [target[j] for (i, j) in path]

	#print(np.array(sample_aligned).T.shape)

	return np.array(sample_aligned).T, np.array(target_aligned).T

def mfcc_to_wav(mfccs, y, sr, output_nm = 'output.wav'):

	# Build reconstruction mappings,
	n_mfcc = mfccs.shape[0]
	n_mel = 128
	dctm = librosa.filters.dct(n_mfcc, n_mel)
	n_fft = 2048
	mel_basis = librosa.filters.mel(sr, n_fft)

	# Empirical scaling of channels to get ~flat amplitude mapping.
	bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

	# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
	recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

	# Impose reconstructed magnitude on white noise STFT.
	excitation = np.random.randn(y.shape[0])
	E = librosa.stft(excitation)
	E = librosa.util.pad_center(E, recon_stft.shape[1], axis=1, mode = 'minimum')
	recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
	# Output
	librosa.output.write_wav(output_nm, recon, sr)


def create_alligned_windows(sample_filename, target_filename, window_size = 0.005):

	sample_mfccs = []
	sample_y_sr = []
	target_mfccs = []
	max_time = round(max(MP3(sample_filename).info.length, MP3(target_filename).info.length),3)
	min_time = min(MP3(sample_filename).info.length, MP3(target_filename).info.length)
	if min_time == MP3(sample_filename).info.length:
		min_file = sample_filename
	else:
		min_file = target_filename
	
	for i in frange(0,max_time,window_size):

		if i <= min_time:
			sample_mfcc, y, sr = gen_mfcc(sample_filename, i, window_size)
			target_mfcc = gen_mfcc(target_filename, i, window_size)[0]
		else:
			if min_file == sample_filename:
				target_mfcc, y, sr = gen_mfcc(target_filename, i, window_size)
				sample_mfcc = np.zeros(target_mfcc.shape)
				y = np.zeros(y.shape)
			else:
				sample_mfcc, y, sr = gen_mfcc(sample_filename, i, window_size)[0]
				target_mfcc = np.zeros(sample_mfcc.shape)

		sample_mfcc,target_mfcc = allign_sample_target(sample_mfcc, target_mfcc, max_size = None)

		sample_mfccs.append(sample_mfcc.T[0])
		sample_y_sr.append((y,sr))
		target_mfccs.append(target_mfcc.T[0])

	sample_mfccs = np.array(sample_mfccs)
	target_mfccs = np.array(target_mfccs)
	
	return sample_mfccs, target_mfccs, sample_y_sr

def produce_wave_parts(samples, sample_y_sr):

	for i in range(samples.shape[0]):

		mfccs = np.array([samples[i]]).T
		y = sample_y_sr[i][0]
		sr = sample_y_sr[i][1]

		mfcc_to_wav(mfccs, y, sr, output_nm = 'testing/'+str(i)+'.wav')

def combine_wav_parts(n):

	combine = 0

	for i in range(n):
		wav = 'testing/'+str(i)+'.wav'
		cmd = 'lame --preset insane %s' % wav
		subprocess.call(cmd, shell=True)

		if combine == 0:
			combine = AudioSegment.from_mp3('testing/'+str(i)+'.mp3')
		else:
			combine += AudioSegment.from_mp3('testing/'+str(i)+'.mp3')

	combine.export("output1.mp3", format="mp3")

	




sample = u"Data/kaggle_cuts/arabic8/Wednesday.mp3"
target = u"Data/kaggle_cuts/english368/Wednesday.mp3"

samplem, targetm, y_sr = create_alligned_windows(sample, target, window_size = 0.01)

# print(np.array([samplem[0]]).T)

# produce_wave_parts(samplem, y_sr)
# #AudioSegment.from_wav("testing/1.wav").export("testing/1.mp3", format="mp3")
# # print(MP3("testing/1.mp3").info.length)
# # AudioSegment.from_mp3("testing/1.mp3")
# combine_wav_parts(152)

print(MP3("output1.mp3").info.length)

# y = y_sr[0][0].shape[0]*samplem.shape[1]
# sr = int(y_sr[0][1]*0.5)

# mfcc_to_wav(samplem, y, sr)

#print(audio.info.length)
#print(gen_mfcc(sample))


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