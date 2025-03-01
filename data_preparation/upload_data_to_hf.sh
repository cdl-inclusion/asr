# run prepare_data.py first to create metadata and bring audio files in correct folder stucture

HF_TOKEN="xxxxx"

REPO="katrintomanek/akan_standard_speech_data_16khz"
LOCAL_DATA_DIR="/xxx/standard_speech/AKAN/16khz"

### upload dataset to github
echo "uploading metadata"
huggingface-cli upload --repo-type=dataset  "${REPO}" "${LOCAL_DATA_DIR}"/metadata.csv metadata.csv --commit-message="Upload metadata" \
    --token="${HF_TOKEN}"



echo "Uploading dev split..."
huggingface-cli upload --repo-type=dataset  "${REPO}" "${LOCAL_DATA_DIR}"/dev dev --commit-message="Upload dev split" \
    --token="${HF_TOKEN}"

echo "Uploading test split..."
huggingface-cli upload --repo-type=dataset  "${REPO}" "${LOCAL_DATA_DIR}"/test test --commit-message="Upload test split" \
    --token="${HF_TOKEN}"

echo "Uploading train0 split..."
huggingface-cli upload --repo-type=dataset  "${REPO}" "${LOCAL_DATA_DIR}"/train_0 train_0 --commit-message="Upload train_0 split" \
    --token="${HF_TOKEN}"

echo "Uploading train1 split..."
huggingface-cli upload --repo-type=dataset  "${REPO}" "${LOCAL_DATA_DIR}"/train_1 train_1 --commit-message="Upload train_1 split" \
    --token="${HF_TOKEN}"

