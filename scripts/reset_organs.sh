#!/bin/bash

for file in leaf/*
do
	mv $file ./
done

for file in flower/*
do
	mv $file ./
done

for file in fruit/*
do
	mv $file ./
done

for file in stem/*
do
	mv $file ./
done

for file in entire/*
do
	mv $file ./
done

for file in branch/*
do
	mv $file ./
done

rmdir leaf/
rmdir flower/
rmdir fruit/
rmdir stem/
rmdir entire/
rmdir branch/
