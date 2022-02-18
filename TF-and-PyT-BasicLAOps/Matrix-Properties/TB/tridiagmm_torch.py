import torch
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


@torch.jit.script
def tridiag_matmul_torch(A,B):
    ret = A@B
    #ret = torch.einsum('ij,jk->ik',A,B)
    return ret

if __name__ == "__main__":

    #Check if MKL is enabled
    print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)


    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 20
    DTYPE = torch.float32

    A = torch.randn([N, N], dtype=DTYPE)
    A = torch.diag(torch.diag(A,1),1)+torch.diag(torch.diag(A))+torch.diag(torch.diag(A,-1),-1)
    B = torch.randn([N, N], dtype=DTYPE)


    for i in range(REPS):
        start = time.perf_counter()
        ret = tridiag_matmul_torch(A,B)
        end = time.perf_counter()
        print("tridiag_matmul_torch : ", end-start) 
    

