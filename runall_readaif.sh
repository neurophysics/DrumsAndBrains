#!/bin/bash
i=1
end=21
while (($i <= $end)); do
    python3 read_aif.py Data/ $i Results/ #~/Neuro/Data/2018-Polyrhythm
    echo python3 read_aif.py Data $i Results/
    wait
    i=$(($i + 1))
done
