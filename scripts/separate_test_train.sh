#!/bin/bash

mkdir testing

mv `grep -l "<LearnTag>Test</LearnTag>" *.xml` ../testing/
mv `ls ../testing/*.xml | sed -e "s/.xml/.jpg/g" | cut -d"/" -f3` ../testing/