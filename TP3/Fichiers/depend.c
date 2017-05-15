#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#define T 10

int A[T][T];

int k = 0;

void tache(int i,int j)
{
  volatile int x = random() % 1000000;
  
  for(int z=0; z < x; z++)
    ;
  
#pragma omp atomic capture
  A[i][j] = k++;
}

int main (int argc, char **argv)
{
  int i, j;
  
#pragma omp parallel
#pragma omp single
  for (i=0; i < T; i++ ){
    for (j=0; j < T; j++ ){
      if(i==0){
        if(j==0){
          #pragma omp task depend(out: A[0][0])
          tache(0,0);
        } else{
          #pragma omp task firstprivate(j) depend(in: A[0][j-1]) depend(out: A[0][j])
          tache(0,j);
	}
      } else{
        if(j==0){
          #pragma omp task firstprivate(i) depend(in: A[i-1][0]) depend(out: A[i][0])
	  tache(i,0);
        } else{
          #pragma omp task firstprivate(i,j) depend(in: A[i-1][j], A[i][j-1]) depend(out: A[i][j])
          tache(i,j);
        }
      }     
    }
  }
    
  for (i=0; i < T; i++ ) {
    puts("");
    for (j=0; j < T; j++ )
      printf(" %2d ",A[i][j]) ;
  }

  return 0;
}
