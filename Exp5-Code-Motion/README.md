## Experiment 5: Code Motion

### Running the experiments

Loop invaraint code motion
```
cd Loop-inv-code/
python loop_inv_tf.py
python loop_inv_torch.py
```

Partial operand access
```
cd partial-op-access/
python partial_op_sum_tf.py
python partial_op_sum_torch.py

python partial_op_prod_tf.py
python partial_op_prod_torch.py
```
#### Operands

> A and B are general square marices of size 3000

> V is a 3x3000


### Performance on Intel Xeon Platinum 8160 CPU

|File | Expression    | TF (naive / recommended)  | PyT (naive / recommended)|
|-----|---------------|---------------------------|--------------|
|loop_inv| for i in range(3): AB + V[i]V[i]<sup>T</sup>  |0.42 / 0.42 |0.42 / 0.41|
|partial_op_sum|(A+B)[2,2] | 0.011 / 6e-4| 0.018 / 2e-3| 
|partial_op_prod|(AB)[2,2] | 0.39 / 2e-3| 0.40 / 3e-3|  


### Performance on Intel Xeon E5-2630 CPU

|File | Expression    | TF (naive / recommended)  | PyT (naive / recommended)|
|-----|---------------|---------------------------|--------------|
|loop_inv| for i in range(3): AB + V[i]V[i]<sup>T</sup>  |0.54 / 0.54 |0.56 / 0.56|
|partial_op_sum|(A+B)[2,2] | 6e-3 / 6e-4| 8e-3 / 2e-3| 
|partial_op_prod|(AB)[2,2] | 0.53 / 7e-4| 0.54 / 3e-3|  
