#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")"
cd "${DIR}/data/" || exit
echo "Downloading auxililary repository containing UniProt data and produced lattice models..."
git clone https://git.wageningenur.nl/lanno001/fret_x_fingerprinting_data.git
echo "Done"
cd ..
