#!/bin/bash
# print md5 checksum and modification date for files
i=1
end=9
cd Data/S01
while (($i <= $end)); do
    cd ../S0$i
    echo S0$i
    files=('clean_data.npz' 'ICA_result.joblib' 'artifact_segments.npy' 'S0$i\_eeg.eeg' 'reject_ICs.txt')
    for file in "${files[@]}"
    do
        m=$(md5 $file | awk '{ print $4 }')
        l=$(ls -l $file | awk '{ print $6 $7 $8 }')
        echo $m $l $file
    done
    echo
    wait
    i=$(($i + 1))
done

i=10
end=21
while (($i <= $end)); do
    cd ../S$i
    echo S$i
    files=('clean_data.npz' 'ICA_result.joblib' 'artifact_segments.npy' 'S0$i\_eeg.eeg' 'reject_ICs.txt')
    for file in "${files[@]}"
    do
        m=$(md5 $file | awk '{ print $4 }')
        l=$(ls -l $file | awk '{ print $6 $7 $8 }')
        echo $m $l $file
    done
    echo
    wait
    i=$(($i + 1))
done
