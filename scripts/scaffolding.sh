#!/bin/bash

nb_images=`ls *.jpg | wc -l`
nb_train=$(($nb_images*80/100))

mkdir training
mkdir testing

ls *.xml | sort -R | tail -$nb_train | while read file; do
	mv `ls $file | sed -e 's/.xml/.jpg/g'` training
	mv $file training
done

ls *.xml | while read file; do
	mv `ls $file | sed -e 's/.xml/.jpg/g'` testing
	mv $file testing
done

shopt -s nullglob globstar

mkdir plants_summary
touch plants_summary/training.data
touch plants_summary/testing.data

mkdir main_vocab
mkdir plants_vocabs
mkdir plants_svm
mkdir results

for file in training/*.xml
do
	FileName=`grep -oPm1 "(?<=<FileName>)[^<]+" $file | cut -d "." -f1`
	ClassId=`grep -oPm1 "(?<=<ClassId>)[^<]+" $file`
	Content=`grep -oPm1 "(?<=<Content>)[^<]+" $file`
	Type=`grep -oPm1 "(?<=<Type>)[^<]+" $file`

	echo $FileName:$ClassId:$Content:$Type >> plants_summary/training.data

	#echo $FileName:${ClassId// /_}:$Content:$Type >> plants_summary/training.data
done

for file in testing/*.xml
do
	FileName=`grep -oPm1 "(?<=<FileName>)[^<]+" $file | cut -d "." -f1`
	ClassId=`grep -oPm1 "(?<=<ClassId>)[^<]+" $file`
	Content=`grep -oPm1 "(?<=<Content>)[^<]+" $file`
	Type=`grep -oPm1 "(?<=<Type>)[^<]+" $file`

	echo $FileName:$ClassId:$Content:$Type >> plants_summary/testing.data

	#echo $FileName:${ClassId// /_}:$Content:$Type >> plants_summary/testing.data
done