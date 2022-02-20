
## Experiment 2: Optimization of Matrix Chains


### Running the experiments

Right to Left

```
cd RtoL/
python rtol_parenthesis_tf.py
python no_parenthesis_tf.py

python rtol_parenthesis_torch.py
python no_parenthesis_torch.py
```

Left to Right

```
cd LtoR/
python ltor_parenthesis_tf.py
python no_parenthesis_tf.py

python ltor_parenthesis_torch.py
python no_parenthesis_torch.py
```

Mixed

```
cd RtoL/
python mixed_parenthesis_tf.py
python no_parenthesis_tf.py

python mixed_parenthesis_torch.py
python no_parenthesis_torch.py
```

#### Operands

> H is a square matrix of size 3000

> x,y are vectors of size 3000


### Performance on Intel Xeon Platinum 8160 CPU

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



### Performance on Intel Xeon E5-2630 CPU

Right to Left Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>Hx | 0.53|0.53| 0.003 
|rtol_parenthesis|H<sup>T</sup>(Hx) | 0.003|0.004| -  |

Left to Right Parenthesization
|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|y<sup>T</sup>H<sup>T</sup>H | 0.003|0.003|  0.003|
|ltor_parenthesis|(y<sup>T</sup>H)<sup>T</sup>H | 0.003|0.004| -|

Mixed Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>yx<sup>T</sup>H | 0.54|0.54|  0.06|
|mixed_parenthesis|(H<sup>T</sup>y)(x<sup>T</sup>H)  | 0.05|0.05| -|
