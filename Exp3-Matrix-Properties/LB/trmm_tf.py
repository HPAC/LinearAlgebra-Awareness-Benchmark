import tensorflow as tf
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


@tf.function
def trmm_tf(A,B):
    op = tf.linalg.LinearOperatorLowerTriangular(A, is_square=True)
    ret = op.matmul(B)
    #ret = A@B
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

    A = tf.linalg.band_part(tf.random.normal([N, N], dtype=DTYPE),-1,0)
    B = tf.random.normal([N, N], dtype=DTYPE)

    #Building Trace
    ret = trmm_tf(A,B)

    for i in range(REPS):
        start = time.perf_counter()
        ret = trmm_tf(A,B)
        end = time.perf_counter()
        print("trmm tf : ", end-start)

