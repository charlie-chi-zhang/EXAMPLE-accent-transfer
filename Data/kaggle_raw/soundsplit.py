import json
from watson_developer_cloud import SpeechToTextV1

service = SpeechToTextV1(
    username='e419556c-7438-4bb7-8ab8-c365eb068e5d',
    password='rVNNXhDSWCA0',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)

# models = service.list_models().get_result()
# print(json.dumps(models, indent=2))

# model = service.get_model('en-US_BroadbandModel').get_result()
# print(json.dumps(model, indent=2))

with open('recordings/english2.mp3','rb') as audio_file:
    data = service.recognize(
            audio=audio_file,
            content_type='audio/mp3',
            timestamps=True,
            word_confidence=True).get_result()

print(data)