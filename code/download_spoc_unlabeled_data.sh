#!/bin/bash

set -x

cd data/spoc-data/train/split/
wget https://www.dropbox.com/s/myswsdqbmm95b1e/spoc_unlabeled_data.tar.gz?dl=1
tar -zxvf spoc_unlabeled_data.tar.gz
