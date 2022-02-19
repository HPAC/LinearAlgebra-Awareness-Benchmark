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
def mc_cse_non_optimized(A,B):
    
    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        #ret = tf.matmul(tf.matmul(A,B),tf.matmul(A,B))
        ret = tf.transpose(A@B)@A@B
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Non Optimized : ", end-start)
    
    return ret

@tf.function
def mc_cse_optimized(A,B):

    start =  tf.timestamp()
    with tf.control_dependencies([start]):
        #tmp1 = A@B
        #ret = tmp1@tmp1
        ret = tf.transpose(A@B)@(A@B)
    with tf.control_dependencies([ret]):
        end =  tf.timestamp()
        tf.print("Optimized : ", end-start)

    
    return ret


A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, n], dtype=DTYPE)



for i in range(reps):
   ret = mc_cse_non_optimized(A,B)
   ret = mc_cse_optimized(A,B)
   tf.print("\n")

