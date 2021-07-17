#!/bin/bash


pip install tqdm
pip install summa
pip install numpy
pip install pandas
pip install sklearn
pip install tensorflow
pip install keras
pip install bert-for-tf2
pip install sentencepiece
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz
tar -xf scibert_scivocab_uncased.tar.gz
mkdir scibert_model
mv scibert_scivocab_uncased/ scibert_model


