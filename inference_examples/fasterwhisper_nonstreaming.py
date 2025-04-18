# Use FasterWhisper for transcription.
# Non-streaming fashion, recording first, then transcribing full recording.
# Audio is captured from microphone using PyAudio.
#
# To run your own models via FasterWhisper, you need to convert them first:
# (set quantization appropriately depending on where you run the model)
#
# ct2-transformers-converter \
#     --model /path/to/model/checkpoint-xyz \
#     --output_dir /tmp/my_converted_model_path \
#     --quantization int8
#
# You can then use the 'output_dir' instead of the 'model_name' when loading the 
# model in the code below.

import numpy as np
import pyaudio
import torch
import sys
from faster_whisper import WhisperModel


def record_audio(duration=5, sample_rate=16000, channels=1, audio_format=pyaudio.paInt16):
    """Record audio for N seconds, then return frames."""
    p = None
    stream = None
    frames = []
    try:
        print(f"Recording {duration} seconds of audio now -- start speaking...")
        p = pyaudio.PyAudio()
        chunk = 1024
        stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
        chunks_to_record = int((sample_rate / chunk) * duration)
        for i in range(chunks_to_record):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            # Show progress of recorded chunks
            if i % 10 == 0:
                sys.stdout.write(f"\rRecording: {i}/{chunks_to_record} chunks")
                sys.stdout.flush()
        print("\nRecording completed.")
        
    except Exception as e:
        print(f"Unexpected error in recording function: {e}")
        return None
    finally:
        # Clean up audio resources
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")
            
        try:
            if p:
                p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

    return frames

def transcribe_audio(audio_data, model_name, word_timestamps=False):
    """Transcribe audio file using Faste Whisper."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(model_name, 
                         device=device, 
                         compute_type=compute_type) 

    segments, _ = model.transcribe(audio_data, 
                                   beam_size=5,
                                   language='en', # if we don't set the language, the model will try to detect it
                                   word_timestamps=word_timestamps, # when True, outputs probabilities on word level
                                   vad_filter=True, # VAD filter often helps with hallucinations by removing silence
                                   )

    full_text = ""
    if word_timestamps:
        for segment in segments:
            for word in segment.words:
                print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word} -- {word}")
    else:
        for segment in segments:
            full_text += segment.text
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
        return full_text


DURATION = 5  # seconds
MODEL_NAME = "tiny"  # or "base", "small" etc
GET_WORD_LEVEL_TIME_STAMPS = False # whether to output confidence scores on word level

# Get audio frames and concat
audio_frames = record_audio(duration=DURATION)
combined_audio_data = b''.join(audio_frames)
audio_np = np.frombuffer(combined_audio_data, dtype=np.int16).astype(np.float32) / 32768.0

# Transcribe audio
_ = transcribe_audio(audio_np, MODEL_NAME, word_timestamps=GET_WORD_LEVEL_TIME_STAMPS)

