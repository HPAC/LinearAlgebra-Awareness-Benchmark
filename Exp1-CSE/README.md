### Experiment 1 : Common Subexpression Elimination

|File | Expression    | TF  | PyT |
|-----|---------------|-----|-----|
|sgemm|A<sup>T</sup>B | 0.53|0.53|  
|cse_add|A<sup>T</sup>B + A<sup>T</sup>B | 0.54|0.54|  
|cse_mul_parenthesis|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| 1.08| 1.09|  
|cse_mul_no_parenthesis|(A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B| 1.61| 1.62|  

> A and B are square matrices of size 3000

#### Running the experiments

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
