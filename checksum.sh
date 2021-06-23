#!/bin/bash
i=1
end=9
cd Data/S01
while (($i <= $end)); do
    cd ../S0$i
    echo S0$i
    md5 clean_data.npz
    md5 ICA_result.joblib
    md5 artifact_segments.npy
    md5 S0$i\_eeg.eeg
    md5 reject_ICs.txt
    echo
    wait
    i=$(($i + 1))
done

i=10
end=21
while (($i <= $end)); do
    cd ../S$i
    echo S$i 
    md5 clean_data.npz
    md5 ICA_result.joblib
    md5 artifact_segments.npy
    md5 S$i\_eeg.eeg
    md5 reject_ICs.txt
    echo
    wait
    i=$(($i + 1))
done
