### Experiment 5: Code Motion

**Performance Measurements:** Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.


|File | Expression    | TF (naive / recommended)  | PyT (naive / recommended)|
|-----|---------------|---------------------------|--------------|
|loop_inv| for i in range(3): AB + V[i]V[i]<sup>T</sup>  |0.54 / 0.54 |0.56 / 0.56|
|partial_op_sum|(A+B)[2,2] | 6e-3 / 6e-4| 8e-3 / 2e-3| 
|partial_op_prod|(AB)[2,2] | 0.53 / 7e-4| 0.54 / 3e-3|  

> A and B are general square marices of size 3000

> V is a 3x3000
