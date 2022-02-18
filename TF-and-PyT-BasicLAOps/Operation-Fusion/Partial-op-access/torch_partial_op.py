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
def actual_expr(A,B):
    ret = (A@B)[2,2]
    #tmp = A@B
    #ret = tmp[2,2]
    return ret

@torch.jit.script
def simplified_expr(A,B):
    ret = torch.dot(A[2],B[:,2])
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)

for i in range(reps):
   start = time.perf_counter()
   ret1 = actual_expr(A,B)
   end = time.perf_counter()
   print("Non Optimized : ", end-start) 

   start = time.perf_counter()
   ret1 = simplified_expr(A,B)
   end = time.perf_counter()
   print("Optimized : ", end-start) 

   #ret2 = simplified_expr(A,B)

   #tf.assert_equal(ret1, ret2)
    
   print("\n")

