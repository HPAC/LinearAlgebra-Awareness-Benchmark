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
def no_parenthesis(H,y):
    ret = tf.transpose(y)@tf.transpose(H)@H 
    return ret

H = tf.random.normal([n, n], dtype=DTYPE)
y = tf.random.normal([n, 1], dtype=DTYPE)


for i in range(reps):

   start = time.perf_counter()
   ret2 = no_parenthesis(H,y)
   end = time.perf_counter()
   print("No Parenthesis : ", end-start) 
    
   tf.print("\n")

