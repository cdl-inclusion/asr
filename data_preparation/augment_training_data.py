# audio data augmentation with pitch shift, time stretch, gaussian noise and room reverberation

import random
import torch
import torchaudio
import librosa
import os
import shutil
import pandas as pd
import numpy as np
import tqdm



class DataAugmentation():
  
  def __init__(self,
               seed = 42,
               sampling_rate=16000,
               probability_pitch_shift=0.5,
               probability_volume_change=0.5,
               probability_time_stretch=0.5,
               probability_noise=0.25,
               probability_room_reverb=0.5,
               verbose=False):

    # audio settings
    self.sampling_rate = 16000

    # seed random for reproducability
    random.seed(seed)

    self.verbose = verbose

    self.probability_pitch_shift = probability_pitch_shift
    self.probability_volume_change = probability_volume_change
    self.probability_time_stretch = probability_time_stretch
    self.probability_noise = probability_noise
    self.probability_room_reverb = probability_room_reverb


    # init room reverb
    # https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html#simulating-room-reverberation    
    rir_sample = torchaudio.utils.download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
    rir_raw, sample_rate = torchaudio.load(rir_sample)
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    self.rir = rir / torch.linalg.vector_norm(rir, ord=2)


  def apply_pitch_shift(self, audio_array):
      pitch_shift = random.choice([-3, -2, -1, 1, 2, 3])
      if self.verbose:
        print('do pitch shift with:', pitch_shift)
      return librosa.effects.pitch_shift(
          y=audio_array, sr=self.sampling_rate, n_steps=pitch_shift)    

  def apply_time_stretch(self, audio_array):
      # stretch_factor = random.uniform(0.95, 1.05)
      stretch_factor = random.choice([0.7, 0.9, 1.0, 1.1, 1.3])
      if self.verbose:
        print('do time stretch with: ', stretch_factor)
      return librosa.effects.time_stretch(
          y=audio_array, rate=stretch_factor)

  def apply_gaussian_noise(self, audio_array):
      # The standard deviation of the Gaussian noise. (higher is more noise)
      # this seems too high
      # noise_level = random.choice([0.01, 0.02])
      #noise_level = random.choice([0.005, 0.01])
      noise_level = 0.005
      if self.verbose:
        print('add noise with: ', noise_level)
      noise = np.random.normal(0, noise_level, audio_array.shape)
      return audio_array + noise

  def apply_room_reverb(self, audio_array):
    if self.verbose:
      print('add reverb')    
    return torchaudio.functional.fftconvolve(audio_array, self.rir)


  def augment_data(self, audio_array):
    filter_applied = False

    if random.random() <= self.probability_pitch_shift:
      filter_applied = True
      audio_array = self.apply_pitch_shift(audio_array)

    if random.random() <= self.probability_volume_change:
      filter_applied = True
      volume_factor = random.uniform(0.8, 1.2)
      if self.verbose:
        print('do volume change with: ', volume_factor)
      audio_array *= volume_factor


    if random.random() <= self.probability_time_stretch:
      filter_applied = True
      audio_array = self.apply_time_stretch(audio_array)

    if random.random() <= self.probability_noise:
      filter_applied = True
      audio_array = self.apply_gaussian_noise(audio_array)


    # bring in torch format for further steps
    audio_array = torch.tensor(
        audio_array, dtype=torch.float32).unsqueeze(0)

    if random.random() <= self.probability_room_reverb:
      filter_applied = True
      audio_array = self.apply_room_reverb(audio_array)

    if filter_applied:
      return audio_array
    else:
      if self.verbose:
        print('>> WARN - no augmentation applied')
      return None


  # convenience helpers
  def load_audio(self, audio_file):
    return librosa.load(audio_file, sr=self.sampling_rate)[0]

  def save_audio(self, audio_array, outout_file):
    print('written to:', outout_file)
    torchaudio.save(outout_file, src=audio_array, sample_rate=self.sampling_rate)



BASE_DATA_DIR = '/Users/katrintomanek/dev/Ghana/speech_data/standard_speech/AKAN/final_split_16khz/data'
AUGMENTED_DATA_DIR = '/Users/katrintomanek/dev/Ghana/speech_data/standard_speech/AKAN/augmented_final_split_16khz'
AUGMENTED_TRAIN_DATA_DIR =  os.path.join(AUGMENTED_DATA_DIR, 'train')
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
os.makedirs(AUGMENTED_TRAIN_DATA_DIR, exist_ok=True)
shutil.copy(os.path.join(BASE_DATA_DIR, 'metadata.csv'), AUGMENTED_DATA_DIR)


## run augmentation
audio_augmenter = DataAugmentation(sampling_rate=16000, verbose=True)

ORIG_TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
for f in [x for x in os.listdir(ORIG_TRAIN_DIR) if not x.startswith('._')]:
    f_path = os.path.join(ORIG_TRAIN_DIR, f)
    audio_array = audio_augmenter.load_audio(f_path)
    augmented_audio_array = audio_augmenter.augment_data(audio_array)
    if augmented_audio_array is not None:
        # safe only files where we have augmentation. 
        outout_file = os.path.join(AUGMENTED_TRAIN_DATA_DIR, f)
        audio_augmenter.save_audio(augmented_audio_array, outout_file)

## filter metadata and overwrite original metadata file
# keep only those entried in the metadata where the file exists
df = pd.read_csv(os.path.join(BASE_DATA_DIR, 'metadata.csv'))
filtered_rows = []
for i, row in df.iterrows():
  audio_file = os.path.join(AUGMENTED_DATA_DIR, row['file_name'])
  if os.path.exists(audio_file):
    filtered_rows.append(row)

df_filtered = pd.DataFrame(filtered_rows)   
df_filtered.to_csv(os.path.join(AUGMENTED_DATA_DIR, 'metadata.csv'), index=False, sep=',')

print('number of augmented training audio files:', len(filtered_rows ))    
print('metadata updated...')
