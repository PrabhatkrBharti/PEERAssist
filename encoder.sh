#!/bin/bash
if [ $# == 0 ]
	then
		echo "ERROR : provide a year ,e.g. ./encoder.sh 2017"
		exit
fi

echo "......GENERATING SECTIONAL-SUMMARY......"
python summaFlatten.py "$1"

echo "......REVIEWS EMBEDDING (USE)......"
python Embed.py "$1"

echo "......PAPERS EMBEDDING (SCIBERT)......"
python paperEmbed.py "$1"
