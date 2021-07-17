#!/bin/bash

if [ $# -lt 1 ]
	then
		echo "ERROR : Provide an year"
		exit
fi
name="./DATA/data/EDATA$1.csv"
rm -rf $name
name="./DATA/data/REV_USE$1.csv"
rm -rf $name
name="./DATA/Dmeta$1.csv"
rm -rf $name
name="./DATA/PMETA$1_SCSUMMA.csv"
rm -rf $name
name="./DATA/flattered/D$1.csv"
rm -rf $name
