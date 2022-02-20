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
def mc_cse_non_optimized(A,B):
    ret = torch.t(torch.t(A)@B)@torch.t(A)@B    
    return ret

@torch.jit.script
def mc_cse_optimized(A,B):
    tmp = torch.t(A)@B
    ret = torch.t(tmp)@tmp
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret = mc_cse_non_optimized(A,B)
   end = time.perf_counter()
   print("Non Optimized : ", end-start)

   start = time.perf_counter()
   ret = mc_cse_optimized(A,B)
   end = time.perf_counter()
   print("Optimized : ", end-start)
   
   print("\n")

