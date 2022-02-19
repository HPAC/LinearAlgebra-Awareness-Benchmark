# LinearAlgebra-Awareness-Benchmark


### Performance

Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.

### C vs Eager vs Graph mode


|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.53|0.53| 0.53| 0.53|0.53|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.57| 1.58| 1.09| 1.09|  

### Experiment 1: Common Sub-Expression Elimination


|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.53|0.53|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.54|0.54|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 1.08| 1.09|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.61| 1.62|  

### Experiment 2: Optimal Parenthesization of Matrix Chain 


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


