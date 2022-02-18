#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <timer.h>
#include "mkl.h"

int main(int argc, char* argv[])
{
  int m, k, n;
  float *A, *B, *C;
  double cs_time = DBL_MAX, dtime, dtime_save = DBL_MAX;

  if (argc < 4) {
    printf("pass me 3 arguments: m k n [name]\n");
    return (-1);
  } else {
    m = atof(argv[1]);
    k = atof(argv[2]);
    n = atof(argv[3]);
  }

  //A = (double*)mkl_malloc(m * k * sizeof(double), 64);
  //B = (double*)mkl_malloc(k * n * sizeof(double), 64);
  //C = (double*)mkl_malloc(m * n * sizeof(double), 64);

  A = (float*)mkl_malloc(m * k * sizeof(float), 32);
  B = (float*)mkl_malloc(k * n * sizeof(float), 32);
  C = (float*)mkl_malloc(m * n * sizeof(float), 32);

  srand48((unsigned)time((time_t*)NULL));

  for (int i = 0; i < m * k; i++) A[i] = drand48();
  for (int i = 0; i < k * n; i++) B[i] = drand48();

  for (int it = 0; it < LAMP_REPS; it++) {

    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    /*printf("A = [\n");*/
    /*for (int i = 0; i < m; i++) {*/
    /*for (int j = 0; j < k; j++)*/
    /*printf(" %lf\t", A[i + j * m]);*/
    /*printf(";\n");*/
    /*}*/
    /*printf("];\n");*/

    /*printf("B = [\n");*/
    /*for (int i = 0; i < k; i++) {*/
    /*for (int j = 0; j < n; j++)*/
    /*printf(" %lf\t", B[i + j * k]);*/
    /*printf(";\n");*/
    /*}*/
    /*printf("];\n");*/

    /*printf("C = [\n");*/
    /*for (int i = 0; i < m; i++) {*/
    /*for (int j = 0; j < n; j++)*/
    /*printf(" %lf\t", C[i + j * m]);*/
    /*printf(";\n");*/
    /*}*/
    /*printf("];\n");*/

    cs_time = cache_scrub();

    dtime = cclock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
    dtime_save = clock_min_diff(dtime_save, dtime);

    /*printf("C2 = [\n");*/
    /*for (int i = 0; i < m; i++) {*/
    /*for (int j = 0; j < n; j++)*/
    /*printf(" %lf\t", C[i + j * m]);*/
    /*printf(";\n");*/
    /*}*/
    /*printf("];\n");*/
    /*printf("using LinearAlgebra\n");*/
    /*printf("C = A * B\n");*/
    /*printf("isapprox(C2, C, atol=1e-4)\n");*/
  }
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);

  if (argv[4]) {
    printf("gemm_explicit_%s_noup;%d;%d;%d;%e;%e\n", argv[4], m, k, n, dtime_save, cs_time);
    printf("gemm_implicit_%s_noup;%d;%d;%d;%e;%e\n", argv[4], m, k, n, dtime_save, cs_time);
  } else {
    printf("gemm_explicit_noup;%d;%d;%d;%e;%e\n", m, k, n, dtime_save, cs_time);
    printf("gemm_implicit_noup;%d;%d;%d;%e;%e\n", m, k, n, dtime_save, cs_time);
  }

  return (0);
}
