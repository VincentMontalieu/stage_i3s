#!/bin/bash

rm -rf plants_summary/
rm -rf main_vocab/
rm -rf plants_vocabs/
rm -rf plants_svm/
rm -rf results/

rm -rf */*.gz

for file in training/*.jpg
do
	mv $file ./
done

for file in training/*.xml
do
	mv $file ./
done

for file in testing/*.jpg
do
	mv $file ./
done

for file in testing/*.xml
do
	mv $file ./
done

rmdir training/ testing/
