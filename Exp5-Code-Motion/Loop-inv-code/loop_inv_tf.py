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
def naive(A,B,V,ret):
    for i in range(3):
        ret = A@B + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
    return ret

@tf.function
def recommended(A,B,V,ret):
    tmp = A@B
    for i in range(3):
        ret = tmp + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
    return ret

A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)
V = tf.random.normal([3, n], dtype=DTYPE)
ret = tf.random.normal([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = naive(A,B,V,ret)
   end = time.perf_counter()
   print("Naive : ", end-start) 

   start = time.perf_counter()
   ret2 = recommended(A,B,V,ret)
   end = time.perf_counter()
   print("Recommended : ", end-start) 
    
   tf.print("\n")

