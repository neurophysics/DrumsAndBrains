#!/bin/bash
i=1
end=21
while (($i <= $end)); do
    echo python -W ignore prepare_filters.py Data/ $i Results/
<<<<<<< HEAD
    python2 -W ignore prepare_filters.py ~/Neuro/Data/2018-Polyrhythm $i Results/
=======
    python2 -W ignore prepare_filters.py Data/ $i Results/
>>>>>>> c9083d7f0ed03acc78c72565dde6f12295b1bd0d
    wait
    i=$(($i + 1))
done
