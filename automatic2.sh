#!/bin/bash
for (( j=1; j<=4; j++ ))
do
	for (( i=1; i<=100; i++ ))
	do
		python3 cov_from_csv.py $j 1 70
		echo "training and test files created"
		./svm-train -s 7 -t 0 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output)
		echo "${conf1}" >> conf_openset.txt
		./svm-train -s 5 -t 0 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output)
		echo "${conf1}" >> conf_oneclass_svm.txt
		echo "Welcome $i times"
	done
	echo "" >> conf_openset.txt
	echo "" >> conf_oneclass_svm.txt
done
echo "With missing classes 5 6" >> conf_openset.txt
echo "With missing classes 5 6" >> conf_oneclass_svm.txt
for (( j=1; j<=4; j++ ))
do
	for (( i=1; i<=100; i++ ))
	do
		python3 cov_from_csv.py $j 1 70 5 6
		echo "training and test files created"
		./svm-train -s 7 -t 0 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output)
		echo "${conf1}" >> conf_openset.txt
		./svm-train -s 5 -t 0 -q training.txt
		conf1=$(./svm-predict test.txt training.txt.model output)
		echo "${conf1}" >> conf_oneclass_svm.txt
		echo "Welcome $i times"
	done
	echo "" >> conf_openset.txt
	echo "" >> conf_oneclass_svm.txt
done


