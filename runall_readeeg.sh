#!/bin/bash
i=15
end=21
while (($i <= $end)); do
    echo python -W ignore read_eeg.py Data/ $i Results/
    python -W ignore read_eeg.py Data/ $i Results/
    wait
    i=$(($i + 1))
done
