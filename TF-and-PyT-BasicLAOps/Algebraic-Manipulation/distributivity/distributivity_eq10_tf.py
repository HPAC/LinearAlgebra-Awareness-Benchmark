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
def lhs(A,H,x):
    ret = A@x - tf.transpose(H)@(H@x)
    return ret

@tf.function
def rhs(A,H,x):
    ret = (A - tf.transpose(H)@H)@x
    return ret

A = tf.random.normal([n, n], dtype=DTYPE)
H = tf.random.normal([n, n], dtype=DTYPE)
x = tf.random.normal([n, 1], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = lhs(A,H,x)
   end = time.perf_counter()
   print("LHS : ", end-start) 

   start = time.perf_counter()
   ret2 = rhs(A,H,x)
   end = time.perf_counter()
   print("RHS : ", end-start) 
    
   tf.print("\n")

