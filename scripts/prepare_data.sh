#!/usr/bin/env bash

python3 -m wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2

gdown https://drive.google.com/u/0/uc?id=15sGIA_9_vd7wiJnm911win_EV4wgtFg
unzip MarkovkaSpeech.zip
mv MarkovkaSpeech data/
rm MarkovkaSpeech.zip