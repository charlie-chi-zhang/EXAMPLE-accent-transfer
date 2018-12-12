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
import os
from process_signal import *

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

	sr = 22050
	n_mfcc=25
	n_mels=128
	n_fft=108

	y, sr = librosa.load(filename, 
						sr = sr, 
						#offset = offset, 
						#duration = duration
						)

	# calculate mfcc
	#Y = librosa.stft(y)
	mfccs = librosa.feature.mfcc(y,
								sr=sr,
								n_mfcc=n_mfcc,
								n_mels=n_mels,
								n_fft=n_fft,
								hop_length=n_fft // 4)

	return mfccs, y, sr

def align_sample_target(sample_mfcc, target_mfcc):

	distance, path = fastdtw(sample_mfcc.T, target_mfcc.T, dist=mfcc_dist)

	sample_aligned = [sample_mfcc.T[i] for (i, j) in path]
	target_aligned = [target_mfcc.T[j] for (i, j) in path]

	return np.array(sample_aligned).T, np.array(target_aligned).T

def mfcc_to_wav(mfccs, y, sr, output_nm = 'output.wav', mult = 1):

	# Build reconstruction mappings,
	n_mfcc = mfccs.shape[0]
	n_mel = 128
	dctm = librosa.filters.dct(n_mfcc, n_mel)
	n_fft = 108
	mel_basis = librosa.filters.mel(sr, n_fft)

	# Empirical scaling of channels to get ~flat amplitude mapping.
	bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

	# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
	recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

	# Impose reconstructed magnitude on white noise STFT.
	excitation = np.random.randn(y.shape[0])
	E = librosa.stft(excitation, n_fft=n_fft)
	print(excitation.shape)
	print(E.shape)
	print(y.shape)
	E = librosa.util.pad_center(E, recon_stft.shape[1], axis=1, mode = 'minimum')
	recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
	recon *= mult
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


def gen_data(sample_accent, target_accent):

	labels = np.array([])
	data = np.array([])
	rootdir = "Data/kaggle_cuts"
	i = 1
	for subdir, dirs, files in os.walk(rootdir):
		accent = subdir.split("/")[-1]
		if accent.startswith(sample_accent):
			print("Processing " + str(i) + "/19 folder")
			i += 1
			for subdir2, dirs2, files2 in os.walk(rootdir):
				accent2 = subdir2.split("/")[-1]
				if accent2.startswith(target_accent):
					for file in files:
						for file2 in files2:
							if file == file2:
								mfcc1, y1, sr1 = gen_mfcc(subdir+"/"+file)
								mfcc2, y2, sr2 = gen_mfcc(subdir2+"/"+file2)
								mfcc1, mfcc2 = align_sample_target(mfcc1, mfcc2)
								if labels.size == 0:
									data = mfcc1.T
									labels = mfcc2.T
								else:
									data = np.vstack((data,mfcc1.T))
									labels = np.vstack((labels,mfcc2.T))
	np.save("kaggle_data.npy", data)
	np.save("kaggle_labels.npy", labels)
	print("Data Collection Finished and Saved!")
	return data, labels

def gen_data_one_word(word, sample_accent, target_accent):
	lim = 50
	labels = []
	data = []
	rootdir = "Data/kaggle_cuts"
	i = 1
	context_size = 30
	accent_cnt = 0
	for subdir, dirs, files in os.walk(rootdir):
		accent = subdir.split("/")[-1]
		if accent.startswith(sample_accent):
			accent_cnt += 1
			if accent_cnt > lim:
				break
			accent2_cnt = 0
			print("Processing " + str(i) + "/a lot of folders")
			i += 1
			for subdir2, dirs2, files2 in os.walk(rootdir):
				accent2 = subdir2.split("/")[-1]
				if accent2.startswith(target_accent):
					accent2_cnt += 1
					if accent2_cnt > lim:
						break
					for file in files:
						for file2 in files2:
							if file.split(".")[0] == word and file2.split(".")[0] == word and file.endswith("wav") and file2.endswith("wav"):
								mfcc1, y1, sr1 = gen_mfcc(subdir+"/"+file)
								mfcc2, y2, sr2 = gen_mfcc(subdir2+"/"+file2)
								mfcc1, mfcc2 = align_sample_target(mfcc1, mfcc2)
								for frame in range(1,mfcc1.T.shape[0]-1):
									if frame < context_size:
										cut = mfcc1.T[:context_size+frame, :]
										cut = np.pad(cut, ((context_size*2+1-len(cut),0),(0,0)), mode = "constant")
									elif mfcc1.T.shape[0] - frame - 1 < context_size:
										cut = mfcc1.T[frame-context_size:, :]
										cut = np.pad(cut, ((0,context_size*2+1-len(cut)),(0,0)), mode = "constant")
									else:
										cut = mfcc1.T[frame-context_size:context_size+frame+1, :]
									cut = cut.reshape(1,cut.shape[0]*cut.shape[1])
									data.append(cut[0])
									label = mfcc2.T[frame-1:frame+2,:] # change for pretraining
									labels.append(label.reshape(1,label.size)[0])
	data = np.array(data)
	labels = np.array(labels)
	print(data.shape)
	print(labels.shape)
	np.save("posttrain"+word+"data.npy", data)
	np.save("posttrain"+word+"labels.npy", labels)
	print("Data Collection Finished and Saved!")
	return data, labels

#gen_data_one_word("Wednesday", "english", "spanish")

# def gen_data_matrix(sample_accent, target_accent, pad = 500):

# 	labels = []
# 	data = []
# 	rootdir = "Data/kaggle_cuts"
# 	i = 1
# 	for subdir, dirs, files in os.walk(rootdir):
# 		accent = subdir.split("/")[-1]
# 		if accent.startswith(sample_accent):
# 			print("Processing " + str(i) + "/19 folder")
# 			i += 1
# 			for subdir2, dirs2, files2 in os.walk(rootdir):
# 				accent2 = subdir2.split("/")[-1]
# 				if accent2.startswith(target_accent):
# 					for file in files:
# 						for file2 in files2:
# 							if file == file2:
# 								mfcc1, y1, sr1 = gen_mfcc(subdir+"/"+file)
# 								mfcc2, y2, sr2 = gen_mfcc(subdir2+"/"+file2)
# 								mfcc1, mfcc2 = align_sample_target(mfcc1, mfcc2)
# 								mfcc1 = np.pad(mfcc1,((0,0),(0,pad-mfcc1.shape[1])), mode = "constant")
# 								mfcc2 = np.pad(mfcc2,((0,0),(0,pad-mfcc2.shape[1])), mode = "constant")
# 								data.append(mfcc1.T)
# 								labels.append(mfcc2.T)
# 	labels = np.array(labels)
# 	data =  np.array(data)
# 	np.save("kaggle_data_matrix.npy", data)
# 	np.save("kaggle_labels_matrix.npy", labels)
# 	print("Data Collection Finished and Saved!")
# 	return data, labels

# def gen_data_matrix_one_word(word, sample_accent, target_accent, pad = 500):

# 	labels = []
# 	data = []
# 	rootdir = "Data/kaggle_cuts"
# 	count = 0
# 	for subdir, dirs, files in os.walk(rootdir):
# 		accent = subdir.split("/")[-1]
# 		if accent.startswith(sample_accent):
# 			count += 1
# 	i = 1
# 	for subdir, dirs, files in os.walk(rootdir):
# 		accent = subdir.split("/")[-1]
# 		if accent.startswith(sample_accent):
# 			print("Processing " + str(i) + "/"+str(count)+" folder")
# 			i += 1
# 			for subdir2, dirs2, files2 in os.walk(rootdir):
# 				accent2 = subdir2.split("/")[-1]
# 				if accent2.startswith(target_accent):
# 					for file in files:
# 						for file2 in files2:
# 							if file.split(".")[0] == word and file2.split(".")[0] == word:
# 								mfcc1, y1, sr1 = gen_mfcc(subdir+"/"+file)
# 								mfcc2, y2, sr2 = gen_mfcc(subdir2+"/"+file2)
# 								mfcc1, mfcc2 = align_sample_target(mfcc1, mfcc2)
# 								mfcc1 = np.pad(mfcc1,((0,0),(0,pad-mfcc1.shape[1])), mode = "constant")
# 								mfcc2 = np.pad(mfcc2,((0,0),(0,pad-mfcc2.shape[1])), mode = "constant")
# 								data.append(mfcc1.T)
# 								labels.append(mfcc2.T)
# 	labels = np.array(labels)
# 	data =  np.array(data)
# 	np.save("kaggle_data_matrix"+word+".npy", data)
# 	np.save("kaggle_labels_matrix"+word+".npy", labels)
# 	print("Data Collection Finished and Saved!")
# 	return data, labels

def gen_graph():

	rootdir = "models"
	for subdir, dirs, files in os.walk(rootdir):
		if "paul" in subdir:
			for f in files:
				if f.endswith(".npy"):
					print(subdir+"/"+f)
	name = input("Which cost curve would you like to see? ")
	dat = np.load(name)[:200]
	print(dat)
	plt.plot(range(len(dat)),dat)
	plt.show()

def process():
	new_sig = SigProc("pauls_rec_e5000_lr0001_100_100_smth_ontrained_3000english419_output_final.wav").logMMSE
	librosa.output.write_wav("plz_jesus.wav", new_sig, 22050, norm=False)

#gen_graph()
#gen_data_one_word("brother", "aenglish", "benglish")
# a = np.load("cost_over_time.npy")
# plt.plot(range(len(a)),a)
# plt.show()

#print(np.load("kaggle_data_matrix.npy").shape)

# print(gen_mfcc("Data/kaggle_cuts/aenglish222/brother.mp3")[0].shape)

# gen_data_matrix("aenglish", "benglish", pad = 1000)


		# for file in files:
		# 	print(file)
# sample = u"Data/kaggle_cuts/arabic8/Wednesday.mp3"
# target = u"Data/kaggle_cuts/english368/Wednesday.mp3"
# m,y,s = gen_mfcc(sample)
# m2,y2,s2 = gen_mfcc(target)
# m,m2 = align_sample_target(m,m2)
# mfcc_to_wav(m2, y2, s2, output_nm = 'output4.wav')

