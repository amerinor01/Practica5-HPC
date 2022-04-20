#include <stdio.h>
#include "cblas.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "memoryfun.h"
#include <omp.h>
#include <mpi.h>

#define SEED 2022
#define TAG 22

#define ALPHA 1
#define BETA 0

#define TO_SECONDS 1000000

#define DEBUG 1

//openMP defines
#define NUM_THREADS 4
//#######################

//MPI defines
#define ROOT 0
//#######################


int main(int argc, char **argv){

int i,j,l;              /* Indices para bucles */
int n;                  /* Tamanyo de las matrices */
int np,mid;             /* numero de procesos (np), identificador del proceso (mid) */
int nlocal;             /* columnas que corresponden a cada proceso */ 

double *A,*B,*C;                    /*Punteros de las matrices globales*/
double *Alocal,*Blocal,*Clocal; 	/*Punteros de las matrices locales*/

/*Metrics vars*/
clock_t inicio, fin;                                                                                                                                                                                        
double  duration;  

srand(SEED);
 
/* Comprobación numero de argumentos correctos. Se pasa m */
if (argc!=2){
   printf("Error de Sintaxis. Uso: mpi_gemm n \n");
   exit(1);
}

/* Lectura de parametros de entrada */
n=atoi(argv[1]); 

/* Inicializar el entorno MPI */
MPI_Init(&argc, &argv);
/* ¿Cuántos procesos somos? */
MPI_Comm_size(MPI_COMM_WORLD, &np);
/* ¿Cuál es mi identificador? */
MPI_Comm_rank(MPI_COMM_WORLD, &mid);
/* n debe ser múltiplo de np. Al menos para empezar. Cuando tengáis más experiencia, esto se puede adaptar. */
nlocal=n/np;  

Alocal=dmatrix(n,nlocal);
Clocal=dmatrix(n,nlocal);
B=dmatrix(n,n);    
   
if (!mid){
    /*Alloc Global non-shared variables*/
    C=dmatrix(n,n);	
    A=dmatrix(n,n);

	/* Relleno de las matrices. Uso de macro propia o memset para inicializar a 0*/
	for (i = 0; i < n; ++i)
        for(j = 0; j < n; ++j)
            M(A,i,j,n) = n*i+j+1; 
	for (i = 0; i < n; ++i)
        for(j = 0; j < n; ++j)
            M(B,i,j,n)=n*n+i*n+j+1;
    
    inicio = clock();
    double st = omp_get_wtime();


#if DEBUG
    printf("MATRIX A\n");
    printMatrix(A,n,n);
    printf("\n");
    printf("MATRIX B\n");
    printMatrix(B,n,n);
    printf("\n");

#endif
}


/* Reserva de espacio para las matrices locales utilizando las rutinas en memoryfun.c */
/* Cada proceso calcula el producto parcial de la matriz */

MPI_Scatter(A, nlocal*n, MPI_DOUBLE, Alocal, n*nlocal, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
MPI_Bcast(B,n*n,MPI_DOUBLE,ROOT,MPI_COMM_WORLD);

/* Cada proceso calcula el producto parcial de la matriz */
memset(Clocal,0.0,nlocal*n*sizeof(double));

#pragma omp parallel private(i)
{  
    #pragma omp for
        for(i = 0; i < nlocal; ++i){

            //printMatrix(Blocal, i,4);
            //printf("\n");
           cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,ALPHA,B,n,&M(Alocal,i,0,nlocal),1,BETA,&M(Clocal,i,0,nlocal),1);
        }
}

//cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nlocal,n,n,ALPHA,Alocal,n,B,n,BETA,Clocal,n);
           
MPI_Gather(Clocal, n*nlocal, MPI_DOUBLE, C, n*nlocal, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 

// Una vez calculada la matriz parcial se envía al proceso maestro, en este caso el 0, 
// para obtener el resultado global.
double end = omp_get_wtime();
fin = clock();



//duration = (double)(fin- inicio)/TO_SECONDS;   
//printf("%7lf\n",duration);

/* Llegado a este punto el proceso 0 ha de tener toda la matriz, por lo que puede imprimirla */
if (!mid){
#if DEBUG
    printf("Matriz C\n");
    printMatrix(C, n, n);
#endif
    duration = (double)(fin- inicio)/TO_SECONDS;   
    printf("%7lf\n",duration);


}
MPI_Finalize();
/* Cerrar el entorno MPI */
return 0;
}

/* 
A =

     1     5     9     13
     2     6    10     14
     3     7    11     15
     4     8    12     16
	
B =

    17    21    25    29
    18    22    26    30
    19    23    27    31
    20    24    28    32


Resultado 


         538(250)         650(260)         762 (270)         874(280)
         612 (618)        740(644)         868 (670)         996(696)
         686 (986)        830 (1028)       974 (1070)        1118(1112)
         760 (1354)       920(1412)        1080(1470)        1240(1528)
		
		*/
		





