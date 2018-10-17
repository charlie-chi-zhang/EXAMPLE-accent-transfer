import json
from watson_developer_cloud import SpeechToTextV1
from pydub import AudioSegment
import os

service = SpeechToTextV1(
    username='e419556c-7438-4bb7-8ab8-c365eb068e5d',
    password='rVNNXhDSWCA0',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)

# models = service.list_models().get_result()
# print(json.dumps(models, indent=2))

# model = service.get_model('en-US_BroadbandModel').get_result()
# print(json.dumps(model, indent=2))


for filename in os.listdir("recordings"):
	if filename.endswith(".mp3"):
		
		mp3file = 'recordings/'+filename

		speaker = filename[:-4]

		dirName = "cuts/"+speaker+"/"

		a = AudioSegment.from_mp3(mp3file)

		with open(mp3file,'rb') as audio_file:
			data = service.recognize(
					audio=audio_file,
					content_type='audio/mp3',
					timestamps=True,
					word_confidence=True).get_result()

		for section in data["results"]:
			if "alternatives" in section:
				alt = section["alternatives"]
				timestamps = alt[0]['timestamps']
				for arr in timestamps:
					word = arr[0]
					strt = arr[1]
					end = arr[2]
					audio_slice = a[strt*1000:end*1000+50]
					if not os.path.exists(dirName):
						os.mkdir(dirName)
						audio_slice.export(dirName + word+".wav", format="wav")
					else:
						audio_slice.export(dirName + word+".wav", format="wav")
