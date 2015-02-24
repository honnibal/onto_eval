#!/usr/bin/env bash

virtualenv env
source env/bin/activate
pip install spacy
wget http://s3-us-west-1.amazonaws.com/media.spacynlp.com/en_data_all-0.6.tgz
tar -xzf en_data_all-0.6.tgz
rm -rf data/deps/
