#!/bin/bash
i=1
end=10
while (($i <= $end)); do
    echo python read_aif.py Data/ $i Results/
    python read_aif.py Data/ $i Results/
    wait
    i=$(($i + 1))
done
