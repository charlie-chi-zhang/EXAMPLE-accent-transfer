from tools.audio_tools import lpc_analysis, lpc_synthesis
from pydub import AudioSegment
import numpy as np
import librosa

# AudioSegment.ffmpeg = r'C:\\ffmpeg-4.0.2-win64-static\\bin\\ffmpeg.exe'

test1 = u"Data/kaggle_cuts/bosnian3/Wednesday.mp3"
test2 = u"Data/kaggle_cuts/english368/Wednesday.mp3"

sound, sr = librosa.load(test2, sr = None)

lpc, per_frame_gain, residual_excitation = lpc_analysis(sound)
resynth = lpc_synthesis(lpc, per_frame_gain)

print(len(lpc))
print(len(residual_excitation))

librosa.output.write_wav('test/test2.wav', resynth, sr)

