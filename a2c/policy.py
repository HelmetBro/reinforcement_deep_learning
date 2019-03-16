import numpy as np
import tensorflow as tf

class Policy():

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        
        #https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks
        gain = np.sqrt(2) #mainly just a safety precaution
        
        batches = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (batches, nh, nw, nc * nstack)

        #give input data its space and normalize it
        inputs = tf.placeholder(tf.uint8, ob_shape)
        normalization = tf.cast(inputs, tf.float32) / 255.0
        
        with tf.variable_scope("model", reuse=reuse):

            #took convo layer values from brandons' ddqn algo, he did enough research

            first_convo = tf.layers.conv2d(
                inputs=normalization, 
                filters=32, 
                kernel_size=8,
                strides=(4, 4), 
                activation=tf.nn.relu,
                kernel_initializer=tf.orthogonal_initializer(gain=gain))

            second_convo = tf.layers.conv2d(
                inputs=first_convo, 
                filters=64, 
                kernel_size=4,
                strides=(2, 2), 
                activation=tf.nn.relu,
                kernel_initializer=tf.orthogonal_initializer(gain=gain))
            
            third_convo = tf.layers.conv2d(
                inputs=second_convo, 
                filters=64, 
                kernel_size=3,
                strides=(1, 1), 
                activation=tf.nn.relu,
                kernel_initializer=tf.orthogonal_initializer(gain=gain))

            flattened_convo_layers = tf.layers.flatten(third_convo)

            connected = tf.layers.dense(
                inputs=flattened_convo_layers, 
                units=512, 
                activation=tf.nn.relu,   
                kernel_initializer=tf.orthogonal_initializer(gain))

            pi = tf.layers.dense(
                inputs=connected, 
                units=ac_space.n, 
                activation=None,   
                kernel_initializer=tf.orthogonal_initializer(1.0))

            vf = tf.layers.dense(
                inputs=connected, 
                units=1, 
                activation=None,   
                kernel_initializer=tf.orthogonal_initializer(1.0))


        v0 = vf[:, 0]
        noise = tf.random_uniform(tf.shape(pi))
        a0 = tf.argmax(pi - tf.log(-tf.log(noise)), 1)

        self.initial_state = []

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {inputs: ob})
            return a, v, []

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {inputs: ob})

        self.X = inputs
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value