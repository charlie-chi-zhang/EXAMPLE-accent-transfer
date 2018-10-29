from pydub import AudioSegment
import numpy as np
import python_speech_features as psf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def get_aligned_mfccs(file_pair):
    """
    Takes a pair of file paths, returns aligned MFCC coefficients for the two
    audio clips
    """
    mfccs = []
    
    for file in file_pair:  
    
        sound = AudioSegment.from_file(file)   
        samples = np.array(list(sound.get_array_of_samples()))
    
        mfcc_samples = psf.mfcc(samples,
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
    
        mfccs.append(mfcc_samples)
    
    sample = mfccs[0]
    target = mfccs[1]
    
    distance, path = fastdtw(sample, target, dist=euclidean)
    
    sample_aligned = [sample[i] for (i, j) in path]
    target_aligned = [target[j] for (i, j) in path]
    
    return (np.array(sample_aligned), np.array(target_aligned))

def reconstruct_from_mfcc(coeffs):
    pass


print(get_aligned_mfccs(("cuts/bosnian3/Wednesday.mp3","cuts/english368/Wednesday.mp3")))