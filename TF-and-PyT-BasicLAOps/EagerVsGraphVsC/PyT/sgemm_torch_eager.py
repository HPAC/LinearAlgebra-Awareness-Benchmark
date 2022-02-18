import torch
import os
import time
#import timeit

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


#Check if MKL is enabled
print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)


#Sets the number of threads used for intraop parallelism on CPU.
torch.set_num_threads(1)


#Problem size
n = 3000
reps = 20


#@tf.function
def gemm_implicit_noup(A, B):
    #C = A @ B
    C = torch.matmul(A,B)
    return C



A = torch.randn([n, n], dtype=torch.float32)
B = torch.randn([n, n], dtype=torch.float32)
#C = torch.randn([n, n], dtype=torch.float32)


for i in range(reps):
   start = time.perf_counter()
   C = gemm_implicit_noup(A,B)
   end = time.perf_counter()
   print(end-start)
