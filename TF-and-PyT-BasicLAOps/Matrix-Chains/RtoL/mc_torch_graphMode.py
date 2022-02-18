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
def mc_non_optimized(A,B,C):
    #ret = A@B@C
    ret = torch.linalg.multi_dot([A,B,C])    
    return ret

@torch.jit.script
def mc_optimized(A,B,C):
    ret = A@(B@C)
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)
C = torch.randn([n, int(n/5)], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret = mc_non_optimized(A,B,C)
   end = time.perf_counter()
   print("Non Optimized : ", end-start)

   start = time.perf_counter()
   ret = mc_optimized(A,B,C)
   end = time.perf_counter()
   print("Optimized : ", end-start)
   
   print("\n")

