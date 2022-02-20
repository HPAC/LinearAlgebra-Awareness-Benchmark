## Performance comparision of Eager mode and Graph mode


### Running the experiments

sgemm
```
cd sgemm/
python sgemm_tf_eager.py
python sgemm_tf_graphMode.py

python sgemm_torch_eager.py
python sgemm_torch_graphMode.py
```

cse
```
cd cse/

python cse_tf_eager.py
python cse_tf_graphMode.py

python cse_torch_eager.py
python cse_torch_graphMode.py
```

#### Operands
> A and B are general square matrices of size 3000

### Performance on Intel Xeon Platinum 8160 CPU

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.39|0.40| 0.40| 0.40|0.40|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.25 | 1.27| 0.78| 0.80| 



### Performance on Intel Xeon E5-2630 CPU

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.52|0.53| 0.53| 0.53|0.53|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.57| 1.58| 1.09| 1.09|  












 
