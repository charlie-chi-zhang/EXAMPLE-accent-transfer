from pydub import AudioSegment
import numpy as np
import python_speech_features as psf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

mccs = []

for file in ["cuts/bosnian3/Wednesday.mp3","cuts/english368/Wednesday.mp3"]:

	sound = AudioSegment.from_file(file)

	# this is an array
	samples = np.array(list(sound.get_array_of_samples()))

	mcc = psf.mfcc(samples,
					samplerate=16000,
					winlen=0.025,
					winstep=0.01,
					numcep=13,
	                nfilt=26,
	                nfft=512,
	                lowfreq=0,
	                highfreq=None,
	                preemph=0.97,
	     			ceplifter=22,
	     			appendEnergy=True)

	mccs.append(mcc)

x = mccs[0]
y = mccs[1]

distance, path = fastdtw(x, y, dist=euclidean)

print(y.shape)

print(path)