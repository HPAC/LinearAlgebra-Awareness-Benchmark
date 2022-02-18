import tensorflow as tf
import os
import time

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
def mc_non_optimized(A,B,C):
    
    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        #ret = tf.matmul(tf.matmul(A,B),C)
        ret = A@B@C
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Non Optimized : ", end-start)
    
    return ret

@tf.function
def mc_optimized(A,B,C):

    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        ret = A@(B@C)
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Optimized : ", end-start)

    
    return ret


A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)
C = tf.random.normal([n, int(n/5)], dtype=DTYPE)


for i in range(reps):
   ret = mc_non_optimized(A,B,C)
   ret = mc_optimized(A,B,C)
   tf.print("\n")

