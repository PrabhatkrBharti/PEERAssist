#!/bin/bash

echo "......EXTRACTING DATASETS......"
./raw/extractor.sh

echo "......MAPPING PAPER , REVIEW AND YEAR......"
python start_mapping.py

echo "......EXTRACTING SENTENCES IN REVIEWS......"
python Flatten.py

echo "......EXTRACTING SENTENCES IN PAPERS......"
python paperFlatten.py





