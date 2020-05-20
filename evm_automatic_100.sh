#!/bin/bash
for (( season=1; season<=4; season++ ))
do
    for ct in $(seq 1 -0.1 0.6)
	do
        echo "${ct}" >> evm_results_100.txt
        for (( i=1; i<=100; i++ ))
        do
            conf1=$(python3 evm_testing.py $season 1 70 $ct)
            echo "${conf1}" >> evm_results_100.txt
        done
    done
    echo "" >> evm_results_100.txt
done
echo "With missing classes 5 6" >> evm_results_100.txt

for (( season=1; season<=4; season++ ))
do
    for ct in $(seq 1 -0.1 0.6)
	do
        echo "${ct}" >> evm_results_100.txt
        for (( i=1; i<=100; i++ ))
        do
            conf1=$(python3 evm_testing.py $season 1 70 $ct 5 6)
            echo "${conf1}" >> evm_results_100.txt
        done
    done
    echo "" >> evm_results_100.txt
done