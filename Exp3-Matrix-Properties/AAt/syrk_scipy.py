import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from scipy import linalg

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def syrk_scipy(A):
    ret = linalg.blas.ssyrk(1.0,A)
    #ret = A@A.T
    return ret



if __name__ == "__main__":


    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 20
    DTYPE = np.float32

    A = np.random.randn(N,N).astype(DTYPE)
    A = A.ravel(order='F').reshape(A.shape, order='F')

    for i in range(REPS):
        start = time.perf_counter()
        ret = syrk_scipy(A)
        end = time.perf_counter()
        print("syrk_scipy: ", end-start)

