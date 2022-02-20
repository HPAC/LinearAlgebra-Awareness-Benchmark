### Performance comparision of Eager mode and Graph mode

**Performance measurements:** Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.53|0.53| 0.53| 0.53|0.53|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.57| 1.58| 1.09| 1.09|  

> A and B are general square matrices of size 3000
