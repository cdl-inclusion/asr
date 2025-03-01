
function resample_folder () {
    echo "Processing: ${INPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    for inpath in ${INPUT_DIR}/*.mp3; do 
        infile=$(basename ${inpath})
        outpath="${OUTPUT_DIR}/$infile"
        ffmpeg -i $inpath -ar 16000 $outpath -hide_banner -loglevel error
    done
}

# INPUT_DIR="/xxx/standard_speech/AKAN/test"
# OUTPUT_DIR="/xxx/standard_speech/AKAN/16khz/test"
# resample_folder    
