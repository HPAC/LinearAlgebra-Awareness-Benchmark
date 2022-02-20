### Experiment 3: Matrix Properties


#### Running the experiments

```
cd AB/
python gemm_scipy.py
python gemm_tf.py
python gemm_torch.py

cd LB/
python trmm_scipy.py
python trmm_tf.py
python trmm_torch.py

cd AAt/
python syrk_scipy.py
python syrk_tf.py
python syrk_torch.py

cd TB/
python tridiagmm_scipy.py
python tridiagmm_tf.py
python tridiagmm_torch.py


cd DB/
python diagmm_scipy.py
python diagmm_tf.py
python diagmm_torch.py
```

Matrix multiplication AB with matrices having special properties

> A : General Matrix

> L : Lower Triangular Matrix

> T : Tridiagonal Matrix

> D : Diagonal Matrix 

#### Performance on Intel Xeon Platinum 8160 CPU

|File | Expression    | SciPy (blas) | TF (matmul)  | TF (optimized) | PyT (matmul)| PyT (optimized) |
|-----|---------------|--------------|--------------|----------------| ---------------|--------------|
|sgemm|AB | 0.40|0.40| - | 0.41|-| 
|trmm|LB|0.24|0.40|n.a|0.40|n.a|
|syrk|AA<sup>T</sup>|0.24|0.41|n.a|0.39|n.a|
|tridiagmm|TB|0.20|0.41|0.02|0.40|n.a|
|diagmm|DB|0.12|0.39|0.018|0.40|n.a|

> A,L,T,D are square matrices of size 3000.

#### Performance on Intel Xeon E5-2630 CPU

|File | Expression    | SciPy (blas) | TF (matmul)  | TF (optimized) | PyT (matmul)| PyT (optimized) |
|-----|---------------|--------------|--------------|----------------| ---------------|--------------|
|sgemm|AB | 0.53|0.53| - | 0.53|-| 
|trmm|LB|0.28|0.53|n.a|0.53|n.a|
|syrk|AA<sup>T</sup>|0.28|0.53|n.a|0.53|n.a|
|tridiagmm|TB|0.20|0.53|0.03|0.53|n.a|
|diagmm|DB|0.12|0.53|0.011|0.53|n.a|
