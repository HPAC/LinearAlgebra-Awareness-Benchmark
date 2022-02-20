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
def time_matvec(A,x):
    ret = tf.linalg.matvec(A,x)
    
    return ret


def time_matmul(A,B):
    ret = tf.matmul(A,B)
    
    return ret

A = tf.random.normal([n, n], dtype=DTYPE)
B = tf.random.normal([n, 1], dtype=DTYPE)
x = tf.random.normal([n], dtype=DTYPE)


for i in range(reps):
   
   start = time.perf_counter()
   ret1 = time_matvec(A,x)
   end = time.perf_counter()
   print("Matvec : ", end-start) 

   start = time.perf_counter()
   ret1 = time_matmul(A,B)
   end = time.perf_counter()
   print("Matmul : ", end-start) 
   tf.print("\n")

