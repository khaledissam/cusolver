CC = /scratch/cuda-7.0/bin/nvcc

all : testCusolver testCuSp testCUSP testCuSpThrust

testCusolver : testCusolver.cu
	$(CC) -o testCusolver testCusolver.cu -lcusolver

testCuSp : testCuSp.cu
	$(CC) -o testCuSp testCuSp.cu -lcusolver -lcusparse

testCUSP : testCUSP.cu
	$(CC) -o testCUSP testCUSP.cu -I/scratch/pkgs/cusplibrary-0.5.0

testCuSpThrust : testCuSpThrust.cu
	$(CC) -o testCuSpThrust testCuSpThrust.cu -lcusolver -lcusparse
