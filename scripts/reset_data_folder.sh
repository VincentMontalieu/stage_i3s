#!/bin/bash

rm -rf plants_summary/
rm -rf main_vocab/
rm -rf plants_vocabs/
rm -rf plants_svm/
rm -rf results/

for file in training/*
do
	mv $file ./
done

for file in testing/*
do
	mv $file ./
done

rmdir training/ testing/
