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
def actual_expr(A,B):
    ret = (A@B)[2,2]
    #tmp = A@B
    #ret = tmp[2,2]
    return ret

@tf.function
def simplified_expr(A,B):
    ret = tf.tensordot(A[2],B[:,2],1)
    return ret

A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = actual_expr(A,B)
   end = time.perf_counter()
   print("Actual : ", end-start) 

   start = time.perf_counter()
   ret2 = simplified_expr(A,B)
   end = time.perf_counter()
   print("Simplified : ", end-start) 
    
   tf.print("\n")

