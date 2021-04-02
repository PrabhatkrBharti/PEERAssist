#!/bin/bash

echo "......GENERATING SECTIONAL-SUMMARY......"
python summaFlatten.py "$1"

echo "......REVIEWS EMBEDDING (USE)......"
python Embed.py "$1"

echo "......PAPERS EMBEDDING (SCIBERT)......"
python paperEmbed.py "$1"
