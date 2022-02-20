import tensorflow as tf
import os
import time
import numpy as np

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


#Check if MKL is enabled
import tensorflow.python.framework as tff
print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)


#Set threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.run_functions_eagerly(False)

#Problem size
n = 3000
reps = 10
DTYPE = tf.float32


@tf.function
def lhs(A,B,C):
    ret = A@B + A@C
    return ret

@tf.function
def rhs(A,B,C):
    ret = A@(B+C)
    return ret

A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)
C = tf.random.normal([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = lhs(A,B,C)
   end = time.perf_counter()
   print("LHS : ", end-start) 

   start = time.perf_counter()
   ret2 = rhs(A,B,C)
   end = time.perf_counter()
   print("RHS : ", end-start) 
    
   tf.print("\n")

