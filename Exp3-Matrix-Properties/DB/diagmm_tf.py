import tensorflow as tf
import os
import time
import numpy as np

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


@tf.function
def diagmm_optimized_tf(A,B):
    #ret = A@B
    ret = tf.linalg.tridiagonal_matmul(A, B, diagonals_format='matrix')
    return ret

@tf.function
def diag_matmul_tf(A,B):
    ret = A@B
    return ret


if __name__ == "__main__":

    #Check if MKL is enabled
    import tensorflow.python.framework as tff
    print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)

    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 20
    DTYPE = tf.float32

    # Run in Graph mode
    tf.config.run_functions_eagerly(False)

    A = tf.random.normal([N, N], dtype=DTYPE)
    A = tf.linalg.band_part(A, 0, 0)
    B = tf.random.normal([N, N], dtype=DTYPE)

    #Building Trace
    ret = diagmm_optimized_tf(A,B)
    check = A@B

    for i in range(REPS):
        start = time.perf_counter()
        ret = diagmm_optimized_tf(A,B)
        end = time.perf_counter()
        print("diagmm_optimized : ", end-start)

        start = time.perf_counter()
        ret1 = diag_matmul_tf(A,B)
        end = time.perf_counter()
        print("diagmm_matmul : ", end-start)

        print('\n')
    
        if not np.allclose(check.numpy(),ret.numpy(),rtol=1e-2,atol=1e-2):
            raise Exception("error in computations")

