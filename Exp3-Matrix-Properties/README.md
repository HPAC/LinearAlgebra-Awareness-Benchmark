### Experiment 3: Matrix Properties

Matrix multiplication AB with matrices having special properties

> A : General Matrix

> L : Lower Triangular Matrix

> T : Tridiagonal Matrix

> D : Diagonal Matrix 

**Performance measurements:** Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.

|File | Expression    | SciPy (blas) | TF (matmul)  | TF (optimized) | PyT (matmul)| PyT (optimized) |
|-----|---------------|--------------|--------------|----------------| ---------------|--------------|
|sgemm|AB | 0.53|0.53| - | 0.53|-| 
|trmm|LB|0.28|0.53|n.a|0.53|n.a|
|syrk|AA<sup>T</sup>|0.28|0.53|n.a|0.53|n.a|
|tridiagmm|TB|0.20|0.53|0.012|0.53|n.a|
|diagmm|DB|0.10|0.53|0.011|0.53|n.a|

> A,L,T,D are square matrices of size 3000.
