### Experiment 1 : Common Subexpression Elimination

**Performance measurements:** Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.53|0.53|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.54|0.54|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 1.08| 1.09|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.61| 1.62|  

> A and B are general matrices of size 3000
