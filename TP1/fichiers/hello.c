#include <stdio.h>
#include <omp.h>

int main()
{
  #pragma omp parallel
  {
    #pragma omp critical
    {
      printf("Bonjour ! Je suis le numero %d\n", omp_get_thread_num());
      printf("Au revoir ! Je suis le numero %d\n", omp_get_thread_num());
    }
  }
  return 0;
}
