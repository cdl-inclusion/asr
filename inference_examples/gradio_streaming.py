# Transcribe each chunk individually in a streaming scenario.
# (Not ideal in a production environment because it transcribes
# each chunk indivudally without context.)


import gradio as gr
from transformers import pipeline
import numpy as np


CHUNK_LENGTH = 2 # seconds
LANGUAGE = "en"
WHISPER_MODEL = "openai/whisper-tiny"

whisper_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL)


def transcribe(state, new_chunk):

  # We are transcribing every chunk individually (which is not ideal in a
  # production environment). We are merely concatenating the transcription
  # outputs.
  prev_text = ""
  if state and "text" in state:
    prev_text = state["text"]
  else:
    state = {"text": ""}

  sr, y = new_chunk

  # Convert to mono if stereo
  if y.ndim > 1:
      y = y.mean(axis=1)  

  # ensures the audio data is in floating-point format with 32-bit precision
  if y.dtype != np.float32:
      y = y.astype(np.float32)

  # normalize amplitude
  if np.max(np.abs(y)) > 1.0:
      y = y / np.max(np.abs(y))        

  prediction = whisper_pipeline(
      {"language": LANGUAGE,
       "sampling_rate": sr,
       "raw": y, })["text"]  

  state["text"] = prev_text + ' ' + prediction
  # Note: we could add the previous audio chunk to the state 
  # for better processing: state["audio"] = y
  print("Full text:", state["text"])
  return state, state["text"]

demo = gr.Interface(
    transcribe,
    inputs=["state", gr.Audio(sources=["microphone"], streaming=True)],
    outputs=["state", "text"],
    live=True,
    stream_every=CHUNK_LENGTH
)

demo.launch(debug=True)