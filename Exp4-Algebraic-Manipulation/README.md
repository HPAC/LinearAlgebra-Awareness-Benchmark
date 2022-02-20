## Experiment 4: Algebraic Manipulations



### Running the experiments

Distributivity
```
cd distributivity/
python distributivity_eq9_tf.py
python distributivity_eq9_torch.py

python distributivity_eq10_tf.py
python distributivity_eq10_torch.py
```

Blocked Matrices
```
cd blocked-matrices/
python blocked_matrices_tf.py
python blocked_matrices_torch.py
```

#### Operands

Distributivity: A and B are General square matrices of size 3000.

Blocked matrices: A<sub>B</sub> = [[A<sub>1</sub>, 0], [0, A<sub>1</sub>]], B<sub>B</sub> = [[B<sub>1</sub>, 0], [0, B<sub>1</sub>]]

### Performance on Intel Xeon Platinum 8160 CPU

|File | Expression    | TF (LHS / RHS)  | PyT (LHS / RHS) |
|-----|---------------|--------------|--------------|
|distributivity_eq9|AB + AC = A(B+C)| 0.78 / 0.40|0.81 / 0.41| 
|distributivity_eq10|Ax - H<sup>T</sup>(Hx) = (A - H<sup>T</sup>H)x| 0.01 / 0.42|0.01 / 0.41| 
|blocked_matrices|A<sub>B</sub>B<sub>B</sub> = [(A<sub>1</sub>B<sub>1</sub>),(A<sub>1</sub>B<sub>1</sub>)]<sup>T</sup>| 0.40 / 0.20|0.40 / 0.20| 

### Performance on Intel Xeon E5-2630 CPU

|File | Expression    | TF (LHS / RHS)  | PyT (LHS / RHS) |
|-----|---------------|--------------|--------------|
|distributivity_eq9|AB + AC = A(B+C)| 1.07 / 0.54|1.05 / 0.53| 
|distributivity_eq10|Ax - H<sup>T</sup>(Hx) = (A - H<sup>T</sup>H)x| 0.05 / 0.51|0.06 / 0.52| 
|blocked_matrices|A<sub>B</sub>B<sub>B</sub> = [(A<sub>1</sub>B<sub>1</sub>),(A<sub>1</sub>B<sub>1</sub>)]<sup>T</sup>| 0.53 / 0.27|0.53 / 0.27| 
