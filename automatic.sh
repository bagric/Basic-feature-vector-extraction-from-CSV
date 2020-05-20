#!/bin/bash
for (( j=1; j<=4; j++ ))
do
	for (( i=1; i<=100; i++ ))
	do
		python3 cov_from_csv.py $j 1 70
		echo "training and test files created"
		./svm-train -t 1 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_firstone.txt
		./svm-train -t 0 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_secondone.txt
		./svm-train -t 1 -d 2 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_thirdone.txt
		./svm-train -c 10000000 -g 0.000000001 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_fourthone.txt
		./svm-train -s 1 -n 0.08 -g 0.000001 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_fifthone.txt
		echo "Welcome $i times"
	done
	echo "" >> conf_firstone.txt
	echo "" >> conf_secondone.txt
	echo "" >> conf_thirdone.txt
	echo "" >> conf_fourthone.txt
	echo "" >> conf_fifthone.txt
done
echo "With missing classes 5 6" >> conf_firstone.txt
echo "With missing classes 5 6" >> conf_secondone.txt
echo "With missing classes 5 6" >> conf_thirdone.txt
echo "With missing classes 5 6" >> conf_fourthone.txt
echo "With missing classes 5 6" >> conf_fifthone.txt
for (( j=1; j<=4; j++ ))
do
	for (( i=1; i<=100; i++ ))
	do
		python3 cov_from_csv.py $j 1 70 5 6
		echo "training and test files created"
		./svm-train -t 1 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_firstone.txt
		./svm-train -t 0 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_secondone.txt
		./svm-train -t 1 -d 2 -c 2 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_thirdone.txt
		./svm-train -c 10000000 -g 0.000000001 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_fourthone.txt
		./svm-train -s 1 -n 0.08 -g 0.000001 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output | sed -r 's/Accuracy\s\=\s([0-9]+\.[0-9]*\%).*/\1/g' | sed -r 's/\./\,/g')
		echo "${conf1}" >> conf_fifthone.txt
		echo "Welcome $i times"
	done
	echo "" >> conf_firstone.txt
	echo "" >> conf_secondone.txt
	echo "" >> conf_thirdone.txt
	echo "" >> conf_fourthone.txt
	echo "" >> conf_fifthone.txt
done


