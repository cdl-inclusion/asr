# prepare AKAN speech data so it can be uploaded to HuggingFace as an Audiofolder
import os
import pandas as pd
import hashlib
import shutil


###### constants for metadata preparation ####

METADATA_BASE_DIR = '/Users/katrintomanek/dev/Ghana/speech_data/standard_speech/AKAN/metadata'

METADATA_DEV_TSV_FILENAME = os.path.join(METADATA_BASE_DIR, 'akan_dev_data.tsv')
METADATA_TEST_TSV_FILENAME = os.path.join(METADATA_BASE_DIR, 'akan_test_data.tsv')
METADATA_TRAIN_TSV_FILENAME = os.path.join(METADATA_BASE_DIR, 'akan_train_data.tsv')

FINAL_METADATA_OUT_FILENAME = os.path.join(METADATA_BASE_DIR, 'metadata.csv')

###### constants for moving files ####

MOVE_TRAIN_FILES = True # whether to do this step
SOURCE_TRAIN_DATA_DIR = '/Users/katrintomanek/dev/Ghana/speech_data/standard_speech/AKAN/train'
DEST_TRAIN_BASE_DIR = '/Users/katrintomanek/dev/Ghana/speech_data/standard_speech/AKAN'

###

def get_utterance_id(row):
  s = row['orig_file_name'] + row['text'] + str(row['speaker_id']) + row['gender']
  return hashlib.md5(str.encode(s)).hexdigest()

def split_train_portion(row):
  # we are using the original "unamed" column for some sort of a random split
  # all above the 75% percentile gets moved to train_1, below to train_0
  # so that train_0 should have just below 10k files, and train_1 the rest, around 2.5k
  cut_off = 2300
  if row['unamed'] < cut_off:
    split = 'train_0'
  else:
    split = 'train_1'
  return split    

def make_train_filename(row):
  split_name = split_train_portion(row)
  f = row['orig_file_name']
  return os.path.join(split_name, f)
  

def prep_metadata(df, split):
  del_cols = ['File No.', 'IMAGE_PATH', 'AUDIO_PATH', 'ORG_NAME', 'PROJECT_NAME', 'Filename']
  new_col_names = ['unamed', 'src_img_url', 'text', 'speaker_id', 'locale', 'gender', 'age', 'device', 'environment', 'year', 'orig_file_name']
  col_order = ['utterance_id', 'unamed', 'speaker_id', 'file_name', 'orig_file_name', 'text', 'locale', 'gender', 'age', 'device', 'environment', 'year', 'src_img_url' ]
  df = df.drop(columns=del_cols)
  df.columns = new_col_names

  df['utterance_id'] = df.apply(get_utterance_id, axis=1)


  # set file_name with relative path
  if split in ['dev', 'test']:
    df['file_name'] = df['orig_file_name'].apply(lambda x: os.path.join(split, x))
  elif split == 'train':
    df['file_name'] = df.apply(make_train_filename, axis=1)
  else:
    raise ValueError('Unknown split:', split)

  # lastly reorder cols and remove 'unamed' col
  df = df[col_order]
  df = df.drop(columns=['unamed'])
  return df



df_dev = pd.read_csv(METADATA_DEV_TSV_FILENAME, sep='\t', encoding="utf-8")
df_dev = prep_metadata(df_dev, split='dev')

df_test = pd.read_csv(METADATA_TEST_TSV_FILENAME, sep='\t', encoding="utf-8")
df_test = prep_metadata(df_test, split='test')

df_train = pd.read_csv(METADATA_TRAIN_TSV_FILENAME, sep='\t', encoding="utf-8")
df_train = prep_metadata(df_train, split='train')

print(len(df_train), len(df_test), len(df_dev))


df = pd.concat([df_dev, df_test, df_train])
df.to_csv(FINAL_METADATA_OUT_FILENAME, index=False, encoding="utf-8")
print('Written metadata file to:', FINAL_METADATA_OUT_FILENAME)
print('number of utterances:', str(len(df)))

#################
# copy files also
#################

def move_data(df, source_dir, dest_dir):
  for i, row in df.iterrows():
    source = os.path.join(source_dir, row['orig_file_name'])
    dest = os.path.join(dest_dir, row['file_name'])
    if not os.path.exists(source):
      print('SKIPPING MISSING FILE:', source)
    else:
        shutil.move(source, dest)

if MOVE_TRAIN_FILES:
    p0 = os.path.join(DEST_TRAIN_BASE_DIR, 'train_0')
    if not os.path.exists(p0):
        os.makedirs(p0)

    p1 = os.path.join(DEST_TRAIN_BASE_DIR, 'train_1')
    if not os.path.exists(p1):
        os.makedirs(p1)        

    print('moving train...')
    move_data(df_train, SOURCE_TRAIN_DATA_DIR, DEST_TRAIN_BASE_DIR)    