# Record first and then transcribe full recording. No streaming.

import gradio as gr
from transformers import pipeline
import numpy as np

LANGUAGE = "en"
WHISPER_MODEL = "openai/whisper-tiny"

whisper_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL)

def transcribe_hf_whisper(audio):
    sr, y = audio
    y = y.astype(np.float32) / 32768.0
    
    prediction = whisper_pipeline(
        {"language": LANGUAGE,
        "sampling_rate": sr,
        "raw": y, })["text"]  
    return prediction

demo = gr.Interface(
    transcribe_hf_whisper,
    gr.Audio(sources="microphone"),
    "text",
)

demo.launch(debug=True)