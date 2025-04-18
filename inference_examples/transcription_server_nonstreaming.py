# Minimal example for a transcription server with a single endpoint for transcription.
# Audio files are expected to be 16kHz mono wav files.
#
# You can request the server to transcribe an audio file by sending a POST request to the /transcribe endpoint with the audio file in the request body.
# Example:
# curl -F "wav=@jfk.wav" http://localhost:8080/transcribe
from faster_whisper import WhisperModel
from flask import Flask, Response, request
import io
import os
import tempfile
import shutil

model_name = "tiny" 
whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
LANGUAGE = 'en'

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # store uploaded file temporarily for processing
    _, tmp_wav_file = tempfile.mkstemp()
    audio = request.files['wav']

    # Note: we are expecting mono 16khz audios
    audio.save(dst=tmp_wav_file)
    print('>> Successfully uploaded audio file %s to %s' %(audio.filename, tmp_wav_file))    
    
    segments, info = whisper_model.transcribe(
        tmp_wav_file,
        language=LANGUAGE, 
        beam_size=5,
        vad_filter=True)

    pred = ''
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        pred += segment.text + ' '
    pred = pred.strip()

    # remove file and copy
    os.remove(tmp_wav_file)    
    return {"response": "success!", "transcript": pred}


if __name__ == '__main__':
    app.run(debug=True, port=8080)