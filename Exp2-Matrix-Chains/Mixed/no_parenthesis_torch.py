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
def no_parenthesis(H,x,y):
    ret = torch.t(H)@y@torch.t(x)@H 
    return ret


@torch.jit.script
def multidot(H,x,y):
    ret = torch.linalg.multi_dot([torch.t(H), y, torch.t(x), H]) 
    return ret


H = torch.randn([n, n], dtype=DTYPE)
x = torch.randn([n, 1], dtype=DTYPE)
y = torch.randn([n, 1], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret = no_parenthesis(H,x,y)
   end = time.perf_counter()
   print("No Parenthesis : ", end-start)


   start = time.perf_counter()
   ret = multidot(H,x,y)
   end = time.perf_counter()
   print("Multidot : ", end-start)

   print("\n")