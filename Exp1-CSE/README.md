## Experiment 1 : Common Subexpression Elimination

### Running the experiments

Addition
```
cd Addition/
python cse_add_tf.py
python cse_add_torch.py
```

Multiplication (Parenthesized)
```
cd Multiplication/Parenthesized
python cse_mul_parenthesis_tf.py
python cse_mul_parenthesis_torch.py
```

Multiplication (Non Parenthesized)
```
cd Multiplication/Non-Parenthesized
python cse_mul_no_parenthesis_tf.py
python cse_mul_no_parenthesis_torch.py
```

#### Operands

A and B are square matrices of size 3000

### Performance on Intel Xeon Platinum 8160 CPU

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.40|0.40|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.40|0.41|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 0.78| 0.80|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.17| 1.15|  


### Performance on Intel Xeon E5-2630 CPU

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.53|0.53|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.54|0.54|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 1.08| 1.09|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.61| 1.62|  
