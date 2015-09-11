#!/bin/bash

mkdir testing
mkdir training

mv `grep -l "<LearnTag>Test</LearnTag>" *.xml` testing/
mv `ls ../testing/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f3` testing/
mv *.xml training/
mv *.jpg training/