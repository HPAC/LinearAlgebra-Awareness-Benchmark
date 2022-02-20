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
def rtol_parenthesis(H,x):
    ret = torch.t(H)@(H@x) 
    return ret


H = torch.randn([n, n], dtype=DTYPE)
x = torch.randn([n, 1], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret = rtol_parenthesis(H,x)
   end = time.perf_counter()
   print("No Parenthesis : ", end-start)
