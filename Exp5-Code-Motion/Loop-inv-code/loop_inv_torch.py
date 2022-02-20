import torch
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'

#Check if MKL is enabled
print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)


#Sets the number of threads used for intraop parallelism on CPU.
torch.set_num_threads(1)

#Problem size
n = 3000
reps = 10
DTYPE = torch.float32


@torch.jit.script
def naive(A,B,V,ret):
    for i in range(3):
        ret = A@B + torch.tensordot(V[i],torch.t(V[i]),dims=0)
    return ret

@torch.jit.script
def recommended(A,B,V,ret):
    tmp = A@B
    ret = torch.empty_like(A)
    for i in range(3):
        ret = tmp + torch.tensordot(V[i],torch.t(V[i]),dims=0)
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)
V = torch.randn([3, n], dtype=DTYPE)
ret = torch.randn([n, n], dtype=DTYPE)

for i in range(reps):
   start = time.perf_counter()
   ret1 = naive(A,B,V,ret)
   end = time.perf_counter()
   print("Naive : ", end-start) 

   start = time.perf_counter()
   ret1 = recommended(A,B,V,ret)
   end = time.perf_counter()
   print("Recommended : ", end-start) 
    
   print("\n")

