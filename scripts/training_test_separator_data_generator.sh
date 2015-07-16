#!/bin/bash

#le dernier sed probleme \r entre windows et linux
grep "<IndividualPlantId>" *.xml | sed "s/<[^>]*>//g" | cut -d":" -f2 |  sort -n |uniq -c | sort -n | sed -e 's/\r//g' > /tmp/res
mkdir training
mkdir testing
while read line
do
	nbr=`echo $line | awk -F" " '{print $1}'`
	id=`echo $line | awk -F" " '{print $2}'`
	if [ $nbr -gt 1 ]
	then
		let b=$(($nbr/2))
		let a=$(($nbr-$b))
		echo "id = "$id
		echo "nbr = "$nbr
		grep '<IndividualPlantId>'$id'</IndividualPlantId>' *.xml > /tmp/g
		mv `head -n $a /tmp/g | cut -d":" -f1 | sed -e 's/.xml/.jpg/g'` training
		mv `head -n $a /tmp/g | cut -d":" -f1` training
		mv `tail -n $b /tmp/g | cut -d":" -f1 | sed -e 's/.xml/.jpg/g'` testing
		mv `tail -n $b /tmp/g | cut -d":" -f1` testing
	else
		echo "id = "$id
		echo "nbr = "$nbr
		grep '<IndividualPlantId>'$id'</IndividualPlantId>' *.xml > /tmp/g
		mv `head -n 1 /tmp/g | cut -d":" -f1 | sed -e 's/.xml/.jpg/g'` training
		mv `head -n 1 /tmp/g | cut -d":" -f1` training

	fi
done < /tmp/res

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
done

for file in testing/*.xml
do
	FileName=`grep -oPm1 "(?<=<FileName>)[^<]+" $file | cut -d "." -f1`
	ClassId=`grep -oPm1 "(?<=<ClassId>)[^<]+" $file`
	Content=`grep -oPm1 "(?<=<Content>)[^<]+" $file`
	Type=`grep -oPm1 "(?<=<Type>)[^<]+" $file`

	echo $FileName:$ClassId:$Content:$Type >> plants_summary/testing.data
done
