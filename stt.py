import argparse
import io
import os
import subprocess

# Imports the Google Cloud client library
# [START migration_import]
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def toText(path):
    # [START speech_quickstart]

    # [END migration_import]

    # Instantiates a client
    # [START migration_client]
    client = speech.SpeechClient()
    # [END migration_client]

    # The name of the audio file to transcribe
    # file_name = os.path.join(
    #     os.path.dirname(__file__),
    #     '.',
    #     path)
    # subprocess.call(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', '22050',
    #                  '-map_metadata', ' -1', '-hide_banner', '-loglevel', 'panic', path])
    # Loads the audio into memory
    with io.open(path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)
    print("here")
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=22050,
        audio_channel_count=1,
        language_code='ko-KR')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    for result in response.results:
        return f'{(result.alternatives[0].transcript)}'
    # [END speech_quickstart]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True)
    args = parser.parse_args()
    print(toText(args.f))
