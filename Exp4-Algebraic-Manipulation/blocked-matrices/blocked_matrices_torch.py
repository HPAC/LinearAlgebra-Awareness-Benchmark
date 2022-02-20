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
nb = int(n/2)

@torch.jit.script
def lhs(A,B):
    ret = A@B
    return ret

@torch.jit.script
def rhs(A1,A2,B1,B2):
    ret = torch.cat((A1@B1, A2@B2),dim=0)
    return ret


A1 = torch.randn([nb, nb], dtype=DTYPE)
A2 = torch.randn([nb, nb], dtype=DTYPE)
A = torch.cat((torch.cat((A1, torch.zeros([nb,nb])) ,dim=1),torch.cat((torch.zeros([nb,nb]),A2) ,dim=1)),dim=0)

B1 = torch.randn([nb, n], dtype=DTYPE)
B2 = torch.randn([nb, n], dtype=DTYPE)
B = torch.cat((B1,B2),dim=0)

for i in range(reps):
   start = time.perf_counter()
   ret1 = lhs(A,B)
   end = time.perf_counter()
   print("LHS : ", end-start) 

   start = time.perf_counter()
   ret1 = rhs(A1,A2,B1,B2)
   end = time.perf_counter()
   print("RHS : ", end-start) 
    
   print("\n")

