#!/bin/bash
# figure 1
echo "figure 1"
for t in real imag abs; do echo $t; ./plotting.py elongated_gaussian -t $t -r translate; ./plotting.py elongated_gaussian -t $t -e 2 -1 -r translate -n; done

# figure 2
echo "figure 2"
./plotting.py earth -t real

# figure 3
echo "figure 3"
for t in real imag abs; do echo $t; ./plotting.py earth -c elongated_gaussian -t $t; ./plotting.py earth -c elongated_gaussian -t $t -e 2 -1; done
