#!/bin/bash
i=1
end=21
while (($i <= $end)); do
    echo python -W ignore prepare_filters.py Data/ $i Results/
    python2 -W ignore prepare_filters.py ~/Neuro/Data/2018-Polyrhythm $i Results/
    wait
    i=$(($i + 1))
done
