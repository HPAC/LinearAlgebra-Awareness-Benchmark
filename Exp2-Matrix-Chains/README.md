
### Performance

Single threaded Execution time (in sec) on Intel AVX-2 x86 CPU. We report the median of 20 stable repetitions.

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


