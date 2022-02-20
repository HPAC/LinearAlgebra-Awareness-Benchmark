### Performance comparision of Eager mode and Graph mode

|File | Expression    | C   | TF (Eager) | PyT (Eager) | TF (Graph) | PyT (Graph) |
|-----|---------------|-----|------------|-------------|------------|-------------|
|sgemm|A<sup>T</sup>B | 0.53|0.53| 0.53| 0.53|0.53|  
|cse|(A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B)| - | 1.57| 1.58| 1.09| 1.09|  

> A and B are general square matrices of size 3000


#### Running the experiments

TF
```
cd TF/
python sgemm_tf_eager.py
python sgemm_tf_graphMode.py

python cse_tf_eager.py
python cse_tf_graphMode.py
```

PyT
```
cd PyT/
python sgemm_torch_eager.py
python sgemm_torch_graphMode.py

python cse_torch_eager.py
python cse_torch_graphMode.py
```
