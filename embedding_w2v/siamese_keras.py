'''

siamese implementation by keras

'''

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

flags = tf.app.flags
FLAGS = flags.FLAGS

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

class Siamese:
