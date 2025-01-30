#!/usr/bin/bash

MODEL_DIR="../model"

download_caffe_models() {
    echo "Downloading GoogleNet (Caffe) model files..."
    wget --no-check-certificate -q --show-progress -O "$MODEL_DIR"/bvlc_googlenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
    wget --no-check-certificate -q --show-progress -O "$MODEL_DIR"/bvlc_googlenet.prototxt https://github.com/BVLC/caffe/raw/master/models/bvlc_googlenet/deploy.prototxt
    wget --no-check-certificate -q --show-progress -O "$MODEL_DIR"/synset_words.txt https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt
}

if [ $? -ne 0 ]; then
    echo "Error: failed to download GoogleNet (Caffe) model files"
    rm -rf "$MODEL_DIR"/*.* # clean up 
    exit 1
fi

download_caffe_models
ls -lh "$MODEL_DIR"