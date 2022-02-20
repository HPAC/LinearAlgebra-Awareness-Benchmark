### Experiment 4: Algebraic Manipulations


**Performance measurements:** Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.


|File | Expression    | TF (LHS / RHS)  | PyT (LHS / RHS) |
|-----|---------------|--------------|--------------|
|distributivity_eq9|AB + AC = A(B+C)| 1.07 / 0.54|1.05 / 0.53| 
|distributivity_eq10|Ax - H<sup>T</sup>(Hx) = (A - H<sup>T</sup>H)x| 0.05 / 0.51|0.06 / 0.52| 
|blocked_matrices|A<sub>B</sub>B<sub>B</sub> = [(A<sub>1</sub>B<sub>1</sub>),(A<sub>1</sub>B<sub>1</sub>)]<sup>T</sup>| 0.53 / 0.27|0.53 / 0.27| 

> Distributivity: A and B are General square matrices of size 3000.

> Blocked matrices: A<sub>B</sub> = [[A<sub>1</sub>, 0], [0, A<sub>1</sub>]], B<sub>B</sub> = [[B<sub>1</sub>, 0], [0, B<sub>1</sub>]]