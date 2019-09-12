#!/bin/bash
i=1
end=21
while (($i <= $end)); do
    python read_aif.py Data/ $i Results/
    echo python read_aif.py ~/Neuro/Data/2018-Polyrhythm $i Results/
    wait
    i=$(($i + 1))
done
