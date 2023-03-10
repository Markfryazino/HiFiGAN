#!/usr/bin/env bash

# download LJSpeech
python3 -m wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2

# download audios for inference
gdown 15sGIA_9_vd7wiJnm911win_EV4wgtFg_
unzip MarkovkaSpeech.zip
mv MarkovkaSpeech data/
rm MarkovkaSpeech.zip

# create mels for inference
python3 compute_mels.py --dataset data/MarkovkaSpeech/
