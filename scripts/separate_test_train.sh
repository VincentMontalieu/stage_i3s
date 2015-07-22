#!/bin/bash

mkdir testing

mv `grep -l "Test" *.xml` ../testing/
mv `ls ../testing/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f2` ../testing/