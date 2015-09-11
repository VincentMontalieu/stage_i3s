#!/bin/bash

mkdir leaf
mkdir flower
mkdir fruit
mkdir stem
mkdir entire
mkdir branch

mv `grep -l "Flower" *.xml` flower/
mv `grep -l "Fruit" *.xml` fruit/
mv `grep -l "Stem" *.xml` stem/
mv `grep -l "Entire" *.xml` entire/
mv `grep -l "Leaf" *.xml` leaf/
mv `grep -l "Branch" *.xml` branch/

mv `ls flower/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` flower/
mv `ls fruit/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` ruit/
mv `ls stem/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` stem/
mv `ls entire/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` entire/
mv `ls leaf/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` leaf/
mv `ls branch/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` branch/