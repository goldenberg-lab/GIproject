#!/bin/bash

# Script to transfer the png whole slide images

# Set up the directories
dir_data=/data/GIOrdinal/data/cidscann_WSI/svs

for i in 1 2 3; do
    dir_i=$dir_data/cidscann_$i
    echo $dir_i
    ascp -v -d -O 33001 -P 33001 -t 8443 -D --mode=send --user=xdryert0 --host=aspera.research.cchmc.org $dir_i /UC_ML_AI/Images/svs
done


echo "~~~ End of transfer_cchmc.sh ~~~"
