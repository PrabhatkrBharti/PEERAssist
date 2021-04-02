#!/bin/bash

tar -xf 2017.tar.xz
tar -xf 2018.tar.xz
tar -xf 2019.tar.xz

tar -xf R2020.tar.xz
tar -xf P2020.tar.xz

mv R2020/* P2020/
rm -rf R2020
mv P2020 2020
