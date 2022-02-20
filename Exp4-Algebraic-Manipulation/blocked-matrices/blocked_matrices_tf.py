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
nb = int(n/2)


@tf.function
def lhs(A,B):
    ret = A@B
    return ret

@tf.function
def rhs(A1,A2,B1,B2):
    ret = tf.concat((A1@B1, A2@B2),0)
    return ret

A1 = tf.random.normal([nb, nb], dtype=DTYPE)
A2 = tf.random.normal([nb, nb], dtype=DTYPE)
A = tf.concat((tf.concat((A1, tf.zeros([nb,nb])) ,1),tf.concat((tf.zeros([nb,nb]),A2) ,1)),0)

B1 = tf.random.normal([nb, n], dtype=DTYPE)
B2 = tf.random.normal([nb, n], dtype=DTYPE)
B = tf.concat((B1,B2),0)


for i in range(reps):
   start = time.perf_counter()
   ret1 = lhs(A,B)
   end = time.perf_counter()
   print("LHS : ", end-start) 

   start = time.perf_counter()
   ret2 = rhs(A1,A2,B1,B2)
   end = time.perf_counter()
   print("RHS : ", end-start) 
    
   tf.print("\n")

