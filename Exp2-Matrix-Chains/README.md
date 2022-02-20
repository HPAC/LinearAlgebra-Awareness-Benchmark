
### Experiment 2: Optimization of Matrix Chains


#### Right to Left Parenthesization

|File | Expression    | TF (matmul)  | PyT (matmul) | PyT (multi_dot)|
|-----|---------------|--------------|--------------|----------------|
|no_parenthesis|H<sup>T</sup>Hx | 0.52|0.53| 0.003 
|rtol_parenthesis|H<sup>T</sup>(Hx) | 0.003|0.004| -  |

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

#### Running the experiments

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
