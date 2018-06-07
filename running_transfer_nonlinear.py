from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
PI = np.pi
### Parameters
num_input_per = 4
num_hidden = num_input_per*2
num_runs = 50 
learning_rate = 0.1
num_epochs = 75000
num_layers = 4
filename_prefix = "paper_nonlinear_transfer_results/"
input_type = "one_hot" # one_hot, orthogonal, gaussian
save_every = 100

###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')
nonlinearity = tf.nn.sigmoid 

for run_i in xrange(num_runs):
    for t2 in ["X0", "XOR", "XOR_of_XORs", "NXOR", "AND"]:
        np.random.seed(run_i)
        tf.set_random_seed(run_i)
        print("Now running aux task: %s, run: %i" % (t2, run_i))
        x1_data, y1_data = datasets.XOR_of_XORs_dataset(num_input_per)
        if t2 == "X0":
            x2_data, y2_data = datasets.X0_dataset(num_input_per)
        elif t2 == "XOR":
            x2_data, y2_data = datasets.XOR_dataset(num_input_per)
        elif t2 == "XOR_of_XORs":
            x2_data, y2_data = datasets.XOR_of_XORs_dataset(num_input_per)
        elif t2 == "NXOR":
            x2_data, y2_data = datasets.NXOR_dataset(num_input_per)
        elif t2 == "AND":
            x2_data, y2_data = datasets.AND_dataset(num_input_per)

        x_data = np.concatenate([np.concatenate([x1_data, np.zeros_like(x2_data)], axis=1),
                                 np.concatenate([np.zeros_like(x1_data), x2_data], axis=1)], axis=0)
        y_data = np.concatenate([np.concatenate([y1_data, np.zeros_like(y2_data)], axis=1),
                                 np.concatenate([np.zeros_like(y1_data), y2_data], axis=1)], axis=0)

        output_size = 1

        num_datapoints = len(x_data)
        batch_size = len(x_data)
        batch_subset = len(x1_data)
        
        input_ph = tf.placeholder(tf.float32, shape=[None, 2*num_input_per])
        target_ph = tf.placeholder(tf.float32, shape=[None, 2*output_size])

        W = tf.get_variable('Wi', shape=[2*num_input_per, num_hidden], initializer=var_scale_init)
        b = tf.get_variable('Bi', shape=[num_hidden,], initializer=tf.zeros_initializer)
        hidden = nonlinearity(tf.matmul(input_ph, W) + b)

        for layer_i in range(1, num_layers-1):
            W = tf.get_variable('Wh%i' % layer_i, shape=[num_hidden, num_hidden], initializer=var_scale_init)
            b = tf.get_variable('B%i' % layer_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
            hidden = nonlinearity(tf.matmul(hidden, W) + b)

        W = tf.get_variable('Wo', shape=[num_hidden, 2*output_size], initializer=var_scale_init)
        b = tf.get_variable('Bo', shape=[2*output_size,], initializer=tf.zeros_initializer)
        output = nonlinearity(tf.matmul(hidden, W) + b)
        
        first_domain_loss = tf.nn.l2_loss(output[:batch_subset, :output_size] - target_ph[:batch_subset, :output_size])
        second_domain_loss = tf.nn.l2_loss(output[batch_subset:, output_size:] - target_ph[batch_subset:, output_size:])
        loss = first_domain_loss + second_domain_loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)	
    
        with tf.Session() as sess:
            def train_epoch():
                for batch_i in xrange(num_datapoints//batch_size):
                    sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :], target_ph: y_data[batch_i:batch_i+batch_size, :]})

            def evaluate():
                curr_loss1 = 0.
                curr_loss2 = 0.
                for batch_i in xrange(num_datapoints//batch_size):
                    curr_loss1 += sess.run(first_domain_loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :], target_ph: y_data[batch_i:batch_i+batch_size, :]})
                    curr_loss2 += sess.run(second_domain_loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :], target_ph: y_data[batch_i:batch_i+batch_size, :]})
                return curr_loss1/num_datapoints, curr_loss2/num_datapoints
            
            sess.run(tf.global_variables_initializer())
                
            with open("results/t2%s_run%i.csv" % (t2, run_i), "w") as fout:
                fout.write("epoch, loss1, loss2\n")
                loss1, loss2 = evaluate()
                print("%i, %f, %f\n" % (0, loss1, loss2))
                fout.write("%i, %f, %f\n" % (0, loss1, loss2))
                for epoch_i in xrange(1, num_epochs + 1):
                    train_epoch()	
                    if epoch_i % save_every == 0:
                        loss1, loss2 = evaluate()
                        print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                        fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))

        tf.reset_default_graph()
