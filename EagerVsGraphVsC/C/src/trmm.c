#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <timer.h>
#include "mkl.h"

int main(int argc, char* argv[])
{
  int m, n;
  float one = 1.0;
  float *A, *B;
  double dtime, dtime_save = DBL_MAX, cs_time = DBL_MAX;

  if (argc < 3) {
    printf("pass me 2 argument: m, n\n");
    return (-1);
  } else {
    m = atof(argv[1]);
    n = atof(argv[2]);
  }

  srand48((unsigned)time((time_t*)NULL));

  A = (float*)mkl_malloc(m * m * sizeof(float), 32);
  B = (float*)mkl_malloc(m * n * sizeof(float), 32);

  for (int i = 0; i < m * n; i++) B[i] = (float)drand48();

    for (int i = 0; i < m; i++){
      for (int j = 0; j < m; j++){
        if (i < j)
          A[i + j * n] = 0.0;
        else
          A[i + j * n] = (float)drand48();
      }
    }

  for (int it = 0; it < LAMP_REPS; it++) {

     printf("A = [\n");
     for (int i = 0; i < m; i++) {
     for (int j = 0; j < m; j++)
     printf(" %lf\t", A[i + j * m]);
     printf("\n");
     }
     printf("];\n");

    printf("B = [\n");
    for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
    printf(" %lf\t", B[i + j * m]);
    printf("\n");
    }
    printf("];\n");

    cs_time = cache_scrub();

    dtime = cclock();

 //printf(" %f\t", one);
 //exit(-1);
    strmm("L", "L", "N", "N", &m, &n, &one, A, &m, B, &m);

    dtime_save = clock_min_diff(dtime_save, dtime);

    /*printf("B2 = [\n");*/
    /*for (int i = 0; i < m; i++) {*/
    /*for (int j = 0; j < n; j++)*/
    /*printf(" %lf\t", B[i + j * n]);*/
    /*printf("\n");*/
    /*}*/
    /*printf("];\n");*/
    /*printf("using LinearAlgebra\n");*/
    /*printf("B = A * B\n");*/
    /*printf("isapprox(B2, B, atol=1e-4)\n");*/
  }
  mkl_free(A);
  mkl_free(B);

  printf("trmm_explicit;%d;%d;%d;%e;%e\n", m, 0, n, dtime_save, cs_time);
  printf("trmm_implicit;%d;%d;%d;%e;%e\n", m, 0, n, dtime_save, cs_time);

  return (0);
}
