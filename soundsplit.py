import json
from watson_developer_cloud import SpeechToTextV1
from pydub import AudioSegment
import os

service = SpeechToTextV1(
    username='4b7d65dd-3483-4ce1-8402-dc00f3485ace',
    password='7ieqG4SXDjMZ',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)

# models = service.list_models().get_result()
# print(json.dumps(models, indent=2))

# model = service.get_model('en-US_BroadbandModel').get_result()
# print(json.dumps(model, indent=2))


os.makedirs("cuts", exist_ok= True)
for filename in os.listdir("recordings"):
	if filename.endswith(".mp3"):
		
		mp3file = 'recordings/'+filename

		speaker = filename[:-4]

		os.makedirs("cuts/"+speaker, exist_ok= True)
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
					audio_slice.export(dirName + word+".wav", format="wav")
