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
tf.config.run_functions_eagerly(True)

#Problem size
n = 3000
reps = 10
DTYPE = tf.float32


#@tf.function
def cse_check_non_optimized(A,B,C,D):
    
    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        tmp1 = tf.add(tf.matmul(A,B),C)
        tmp2 = tf.add(tf.matmul(A,B),D)
        ret = tf.add(tmp1, tmp2)
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Non Optimized : ", end-start)
    
    return ret

#@tf.function
def cse_check_optimized(A,B,C,D):

    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        tmp1 = tf.matmul(A,B)
        tmp2 = tf.add(tmp1,C)
        tmp3 = tf.add(tmp1,D)
        ret = tf.add(tmp2, tmp3)
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Optimized : ", end-start)

    
    return ret


A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)
C = tf.random.normal([n, n], dtype=DTYPE)
D = tf.random.normal([n, n], dtype=DTYPE)


for i in range(reps):
   ret = cse_check_non_optimized(A,B,C,D)
   ret = cse_check_optimized(A,B,C,D)
   tf.print("\n")

