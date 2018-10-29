import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

import os
import wave

import pylab

name = "cuts/czech5/Wednesday"

def graph_spectrogram(wav_file, dirName):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(dirName+'.png')
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

os.makedirs("spectrograms", exist_ok = True)
for subdir, dirs, files in os.walk("cuts"):
    if len(subdir.split("/")) < 2:
        continue
    speaker = subdir.split("/")[1]
    os.makedirs("spectrograms/"+speaker, exist_ok = True)
    for sound in files:
        if sound.endswith(".wav"):
            word = sound.split(".")[0]

            dirName = "spectrograms/"+speaker+"/"+word

            # sound = AudioSegment.from_mp3(subdir + "/" + sound)
            # sound.export(subdir + "/" + word + ".wav", format="wav")

            graph_spectrogram(subdir + "/" + word + ".wav", dirName)


