#!/bin/bash
i=1
end=21
while (($i <= $end)); do
    echo python3 -W ignore read_eeg.py Data/ $i Results/
    python3 -W ignore read_eeg.py Data/ $i Results/
    wait
    i=$(($i + 1))
done
