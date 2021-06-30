#!/bin/bash
# print md5 checksum and modification date for files
set -u
i=1
end=0
cd ~/Neuro/Data/2018-Polyrhythm/S01
while ((${i} <= ${end})); do
    cd ../S0${i}
    echo S0${i}
    files=('clean_data.npz' 'ICA_result.joblib' 'artifact_segments.npy' "S0${i}_eeg.eeg" 'reject_ICs.txt')
    for file in "${files[@]}"
    do
        #m=$(md5sum $file | awk '{ print $4 }')
        l=$(ls -l $file | awk '{ print $6 $7 $8 }')
	echo $(md5sum $file)
        echo $l
	echo ''
    done
    echo
    wait
    i=$(($i + 1))
done

i=21
end=21
cd ~/Neuro/Data/2018-Polyrhythm/S01
while ((${i} <= ${end})); do
    cd ../S${i}
    echo S${i}
    files=('clean_data.npz' 'ICA_result.joblib' 'artifact_segments.npy' "S${i}_eeg.eeg" 'reject_ICs.txt')
    for file in "${files[@]}"
    do
        #m=$(md5sum $file | awk '{ print $4 }')
        l=$(ls -l $file | awk '{ print $6 $7 $8 }')
	echo $(md5sum $file)
        echo $l
	echo ''
    done
    echo
    wait
    i=$(($i + 1))
done
