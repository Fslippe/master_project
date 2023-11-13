import numpy as np 
import tensorflow as tf 


tf.config.threading.set_inter_op_parallelism_threads(1)

x = np.array([1,1,1])
patches = tf.image.extract_patches(images=x[0],
                                            sizes=[1,128,128, 1],
                                            strides=[1,128,128, 1],
                                            rates=[1,1,1, 1],
                                            padding="VALID")

                                            