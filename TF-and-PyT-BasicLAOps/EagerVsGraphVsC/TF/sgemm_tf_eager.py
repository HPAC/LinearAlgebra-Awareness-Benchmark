import tensorflow as tf
import os
import time
#import timeit

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


#Check if MKL is enabled
import tensorflow.python.framework as tff
print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)


#Set threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

#Problem size
n = 3000
reps = 20


#@tf.function
def gemm_implicit_noup(A, B, C):
    #C = A @ B
    C = tf.matmul(A,B)
    return C



A = tf.random.normal([n, n], dtype=tf.float32)
B = tf.random.normal([n, n], dtype=tf.float32)
C = tf.random.normal([n, n], dtype=tf.float32)

#print("Time : ", timeit.timeit(lambda: gemm_implicit_noup(A,B,C),number=10))

for i in range(reps):
   start = time.perf_counter()
   C = gemm_implicit_noup(A,B,C)
   end = time.perf_counter()
   print(end-start)

