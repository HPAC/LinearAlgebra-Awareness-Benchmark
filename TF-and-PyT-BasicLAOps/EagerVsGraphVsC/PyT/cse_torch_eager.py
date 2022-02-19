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


#@torch.jit.script
def actual_expr(A,B):
    ret = torch.t(torch.t(A)@B)@(torch.t(A)@B)
    return ret

#@torch.jit.script
def simplified_expr(A,B):
    tmp = torch.t(A)@B
    ret = torch.t(tmp)@tmp
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = actual_expr(A,B)
   end = time.perf_counter()
   print("CSE Non Optimized : ", end-start) 

   start = time.perf_counter()
   ret1 = simplified_expr(A,B)
   end = time.perf_counter()
   print("CSE Optimized : ", end-start) 

   #ret2 = simplified_expr(A,B)

   #tf.assert_equal(ret1, ret2)
    
   print("\n")

