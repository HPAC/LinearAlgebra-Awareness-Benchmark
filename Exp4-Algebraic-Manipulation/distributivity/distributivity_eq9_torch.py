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
def lhs(A,B,C):
    ret = A@B + A@C
    return ret

@torch.jit.script
def rhs(A,B,C):
    ret = A@(B+C)
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)
C = torch.randn([n, n], dtype=DTYPE)

for i in range(reps):
   start = time.perf_counter()
   ret1 = lhs(A,B,C)
   end = time.perf_counter()
   print("LHS : ", end-start) 

   start = time.perf_counter()
   ret1 = rhs(A,B,C)
   end = time.perf_counter()
   print("RHS : ", end-start) 
    
   print("\n")

