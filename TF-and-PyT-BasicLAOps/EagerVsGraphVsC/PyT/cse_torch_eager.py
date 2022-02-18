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



def cse_check_non_optimized(A,B,C,D):
    
    start = time.perf_counter()
    tmp1 = torch.add(torch.matmul(A,B),C)
    tmp2 = torch.add(torch.matmul(A,B),D)
    ret = torch.add(tmp1, tmp2)
    end = time.perf_counter()

    print("Non Optimized : ", end-start)
    
    return ret


def cse_check_optimized(A,B,C,D):

    start = time.perf_counter()
    tmp1 = torch.matmul(A,B)
    tmp2 = torch.add(tmp1,C)
    tmp3 = torch.add(tmp1,D)
    ret = torch.add(tmp2, tmp3)
    end = time.perf_counter()

    print("Optimized : ", end-start)

    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)
C = torch.randn([n, n], dtype=DTYPE)
D = torch.randn([n, n], dtype=DTYPE)


for i in range(reps):
   ret = cse_check_non_optimized(A,B,C,D)
   ret = cse_check_optimized(A,B,C,D)
   print("\n")

