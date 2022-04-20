CC = mpicc
C_OPTIMIZED_FLAGS = -O3 -match=native
CFLAGS= -Iinclude/ -fopenmp
all: mpi_gemm

mpi_gemm: 
	$(CC) -c src/memoryfun.c -o lib/memoryfun.o  $(CFLAGS) 
	$(CC) -c src/esqueleto.c -o lib/esqueleto.o  $(CFLAGS) 
	$(CC) -o opt/esqueleto lib/esqueleto.o lib/memoryfun.o -lcblas -fopenmp


