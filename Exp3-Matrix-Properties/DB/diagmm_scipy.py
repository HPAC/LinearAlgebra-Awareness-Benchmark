import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from scipy import linalg

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def diagmm_scipy(A,B,C):
    diag = np.diag(A)
    for i,a in enumerate(diag):
        C[i] = linalg.blas.sscal(a,B[i]) 
    return C



if __name__ == "__main__":


    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 20
    DTYPE = np.float32

    A = np.random.randn(N,N).astype(DTYPE)
    A = np.diag(np.diag(A))
    A = A.ravel(order='F').reshape(A.shape, order='F')
    B = np.random.randn(N,N).astype(DTYPE)
    B = B.ravel(order='F').reshape(B.shape, order='F')
    C = np.random.randn(N,N).astype(DTYPE)
    C = C.ravel(order='F').reshape(C.shape, order='F')

    check = A@B

    for i in range(REPS):
        start = time.perf_counter()
        ret = diagmm_scipy(A,B,C)
        end = time.perf_counter()
        print("diagmm_scipy: ", end-start)
        
        if not np.allclose(check,ret,rtol=1e-2,atol=1e-2):
            raise Exception("error in computations")
