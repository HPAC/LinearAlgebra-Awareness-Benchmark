# LinearAlgebra-Awareness-Benchmark

## Prerequisites

  TensorFlow and PyTorch with MKL support
  
  > [**TensorFlow: Installation instructions**](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html)

  > [**PyTorch: Installation instructions**](https://pytorch.org/get-started/locally/)

## Running the Experiments

> More details in the respective experiment folders



## Performance on Intel Xeon Platinum 8160 CPU:

  Number of Threads: 1 <br/>
  Reporting statistic: Minimum of 20 repetitions. <br/>
  TensorFlow Version: 2.7.0 <br/>
  PyTorch Version: 1.10.0 <br/>
  MKL: Intel OneAPI 2022.0.0 <br/>

### Performance comparision: Eager mode vs Graph mode

**Operands**
>  A and B are general square matrices of size 3000

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.39|0.40| 0.40| 0.40|0.40|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.25 | 1.27| 0.78| 0.80| 



### Experiment 1 : Common Subexpression Elimination

**Operands**
>  A and B are general square matrices of size 3000

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.40|0.40|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.40|0.41|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 0.78| 0.80|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.17| 1.15|  


### Experiment 2: Optimization of Matrix Chains

**Operands**

> H is a square matrix of size 3000

> x,y are vectors of size 3000

Right to Left Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>Hx | 0.40|0.41| 0.006 
|rtol_parenthesis|H<sup>T</sup>(Hx) | 0.006|0.004| -  |

Left to Right Parenthesization
|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|y<sup>T</sup>H<sup>T</sup>H | 0.006|0.005|  0.005|
|ltor_parenthesis|(y<sup>T</sup>H)<sup>T</sup>H | 0.006|0.005| -|

Mixed Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>yx<sup>T</sup>H | 0.41|0.40|  0.01|
|mixed_parenthesis|(H<sup>T</sup>y)(x<sup>T</sup>H)  | 0.01|0.01| -|


### Experiment 3: Matrix Properties

Matrix multiplication AB with matrices having special properties

> A : General Matrix

> L : Lower Triangular Matrix

> T : Tridiagonal Matrix

> D : Diagonal Matrix 

**Operands** :  A,L,T,D are square matrices of size 3000.

|File | Expression    | SciPy (blas) | TF (matmul)  | TF (optimized) | PyT (matmul)| PyT (optimized) |
|-----|---------------|--------------|--------------|----------------| ---------------|--------------|
|sgemm|AB | 0.40|0.40| - | 0.41|-| 
|trmm|LB|0.24|0.40|n.a|0.40|n.a|
|syrk|AA<sup>T</sup>|0.24|0.41|n.a|0.39|n.a|
|tridiagmm|TB|0.20|0.41|0.02|0.40|n.a|
|diagmm|DB|0.12|0.39|0.018|0.40|n.a|


### Experiment 4: Algebraic Manipulations

**Operands**:

> Distributivity: A and B are General square matrices of size 3000.

> Blocked matrices: A<sub>B</sub> = [[A<sub>1</sub>, 0], [0, A<sub>1</sub>]], B<sub>B</sub> = [[B<sub>1</sub>, 0], [0, B<sub>1</sub>]]

|File | Expression    | TF (LHS / RHS)  | PyT (LHS / RHS) |
|-----|---------------|--------------|--------------|
|distributivity_eq9|AB + AC = A(B+C)| 0.78 / 0.40|0.81 / 0.41| 
|distributivity_eq10|Ax - H<sup>T</sup>(Hx) = (A - H<sup>T</sup>H)x| 0.01 / 0.42|0.01 / 0.41| 
|blocked_matrices|A<sub>B</sub>B<sub>B</sub> = [(A<sub>1</sub>B<sub>1</sub>),(A<sub>1</sub>B<sub>1</sub>)]<sup>T</sup>| 0.40 / 0.20|0.40 / 0.20| 


### Experiment 5: Code Motion

**Operands**:

> A and B are general square marices of size 3000

> V is a 3x3000


|File | Expression    | TF (naive / recommended)  | PyT (naive / recommended)|
|-----|---------------|---------------------------|--------------|
|loop_inv| for i in range(3): AB + V[i]V[i]<sup>T</sup>  |0.42 / 0.42 |0.42 / 0.41|
|partial_op_sum|(A+B)[2,2] | 0.011 / 6e-4| 0.018 / 2e-3| 
|partial_op_prod|(AB)[2,2] | 0.39 / 2e-3| 0.40 / 3e-3|  






