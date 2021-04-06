#!/bin/bash

echo "......EXTRACTING DATASETS......"
tar -xf ./raw/2017.tar.xz
mv ./2017 ./raw/2017
tar -xf ./raw/2018.tar.xz
mv ./2018 ./raw/2018
tar -xf ./raw/2019.tar.xz
mv ./2019 ./raw/2019
tar -xf ./raw/P2020.tar.xz
mv ./P2020 ./raw/P2020
mkdir ./raw/2020
cp ./raw/P2020/* ./raw/2020/
rm -rf ./raw/P2020
tar -xf ./raw/R2020.tar.xz
cp ./R2020/* ./raw/2020/
rm -rf R2020
