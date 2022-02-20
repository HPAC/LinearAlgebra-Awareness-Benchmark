# LinearAlgebra-Awareness-Benchmark


### Performance comparision: Eager mode vs Graph mode

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.53|0.53| 0.53| 0.53|0.53|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.57| 1.58| 1.09| 1.09|  

> A and B are general square matrices of size 3000

### Experiment 1 : Common Subexpression Elimination

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.53|0.53|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.54|0.54|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 1.08| 1.09|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.61| 1.62|  

> A and B are square matrices of size 3000


### Experiment 2: Optimization of Matrix Chains

#### Right to Left Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>Hx | 0.52|0.53| 0.003 
|rtol_parenthesis|H<sup>T</sup>(Hx) | 0.003|0.004| -  

#### Left to Right Parenthesization
|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|y<sup>T</sup>H<sup>T</sup>H | 0.003|0.003|  0.003|
|ltor_parenthesis|(y<sup>T</sup>H)<sup>T</sup>H | 0.003|0.004| -|

#### Mixed Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>yx<sup>T</sup>H | 0.54|0.54|  0.07|
|mixed_parenthesis|(H<sup>T</sup>y)(x<sup>T</sup>H)  | 0.05|0.05| -|


> H is a square matrix of size 3000

> x,y are vectors of size 3000

### Experiment 3: Matrix Properties

Matrix multiplication AB with matrices having special properties

> A : General Matrix

> L : Lower Triangular Matrix

> T : Tridiagonal Matrix

> D : Diagonal Matrix 


|File | Expression    | SciPy (blas) | TF (matmul)  | TF (optimized) | PyT (matmul)| PyT (optimized) |
|-----|---------------|--------------|--------------|----------------| ---------------|--------------|
|sgemm|AB | 0.53|0.53| - | 0.53|-| 
|trmm|LB|0.28|0.53|n.a|0.53|n.a|
|syrk|AA<sup>T</sup>|0.28|0.53|n.a|0.53|n.a|
|tridiagmm|TB|0.20|0.53|0.012|0.53|n.a|
|diagmm|DB|0.10|0.53|0.011|0.53|n.a|

> A,L,T,D are square matrices of size 3000.

### Experiment 4: Algebraic Manipulations


|File | Expression    | TF (LHS / RHS)  | PyT (LHS / RHS) |
|-----|---------------|--------------|--------------|
|distributivity_eq9|AB + AC = A(B+C)| 1.07 / 0.54|1.05 / 0.53| 
|distributivity_eq10|Ax - H<sup>T</sup>(Hx) = (A - H<sup>T</sup>H)x| 0.05 / 0.51|0.06 / 0.52| 
|blocked_matrices|A<sub>B</sub>B<sub>B</sub> = [(A<sub>1</sub>B<sub>1</sub>),(A<sub>1</sub>B<sub>1</sub>)]<sup>T</sup>| 0.53 / 0.27|0.53 / 0.27| 

> Distributivity: A and B are General square matrices of size 3000.

> Blocked matrices: A<sub>B</sub> = [[A<sub>1</sub>, 0], [0, A<sub>1</sub>]], B<sub>B</sub> = [[B<sub>1</sub>, 0], [0, B<sub>1</sub>]]

### Experiment 5: Code Motion


|File | Expression    | TF (naive / recommended)  | PyT (naive / recommended)|
|-----|---------------|---------------------------|--------------|
|loop_inv| for i in range(3): AB + V[i]V[i]<sup>T</sup>  |0.54 / 0.54 |0.56 / 0.56|
|partial_op_sum|(A+B)[2,2] | 6e-3 / 6e-4| 8e-3 / 2e-3| 
|partial_op_prod|(AB)[2,2] | 0.53 / 7e-4| 0.54 / 3e-3|  

> A and B are general square marices of size 3000

> V is a 3x3000




