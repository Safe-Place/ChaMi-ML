#!/bin/bash

# Anchor
gdown https://drive.google.com/uc?id=13EXeY3Q29S5kdRU7L1x3KCIr9l7I8toK

# Negative
gdown https://drive.google.com/uc?id=1SpFf-7W9YYCFM1zIUNNH-2Fcx8pevWDY

# Positive
gdown https://drive.google.com/uc?id=1U3e8UA7wDq_myqMdir6Zz9Ea_N3K-EUT

# Unzip all datasets simultaneously
for z in *.zip; do unzip "$z" -d ../Data; done