from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
import os
PI = np.pi
### Parameters
num_input_per = 4
num_hidden = 40
num_task_sessions = 4 # how many times will tasks be repeated
num_runs = 10 
learning_rate = 0.005
num_epochs = 20000
num_layers = 4
init_mult = 1.
output_dir = "sequence_results_nh_%i_lr_%.4f_im_%.2f/" %(num_hidden, learning_rate, init_mult)
save_every = 20
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
continue_training_all = False # If train_sequentially, whether to continue training on task 2 while training task 1
batch_size = 4
early_stopping_thresh = 0.005

###
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=init_mult, mode='FAN_AVG')
if tf_pm:
    nonlinearity = tf.nn.tanh
else:
    nonlinearity = tf.nn.sigmoid 


for input_shared in [False]:#, True]:
    for run_i in range(num_runs):
        for task in ["XOR", "AND", "XOR_of_XORs", "OR"]:
            np.random.seed(run_i)
            tf.set_random_seed(run_i)
            filename_prefix = "task%s_sharedinput%s_run%i" %(task, str(input_shared), run_i)
            print("Now running %s" % filename_prefix)
            if task == "X0":
                x1_data, y1_data = datasets.X0_dataset(num_input_per)
            elif task == "XOR":
                x1_data, y1_data = datasets.XOR_dataset(num_input_per)
            elif task == "XOR_of_XORs":
                x1_data, y1_data = datasets.XOR_of_XORs_dataset(num_input_per)
            elif task == "NXOR":
                x1_data, y1_data = datasets.NXOR_dataset(num_input_per)
            elif task == "AND":
                x1_data, y1_data = datasets.AND_dataset(num_input_per)
            elif task == "OR":
                x1_data, y1_data = datasets.OR_dataset(num_input_per)
            elif task == "parity":
                x1_data, y1_data = datasets.parity_dataset(num_input_per)

            x_l, x_w = x1_data.shape
            y_l, y_w = y1_data.shape
            if input_shared:
                x_data = np.zeros([x_l*num_task_sessions, x_w]) 
            else:
                x_data = np.zeros([x_l*num_task_sessions, x_w*num_task_sessions]) 
            y_data = np.zeros([y_l*num_task_sessions, y_w*num_task_sessions]) 

            for task_i in range(num_task_sessions):
                if input_shared:
                    x_data[x_l*task_i:x_l*(task_i+1), :] = x1_data
                else:
                    x_data[x_l*task_i:x_l*(task_i+1),  x_w*task_i:x_w*(task_i+1)] = x1_data
                y_data[y_l*task_i:y_l*(task_i+1),  y_w*task_i:y_w*(task_i+1)] = y1_data

            if tf_pm:
                x_data = 2*x_data - 1
                y_data = 2*y_data - 1

            output_size = 1

            num_datapoints = len(x_data)
            batch_subset = len(x1_data)
            
            if input_shared:
                N1 = num_input_per
            else:
                N1 = num_task_sessions*num_input_per
            N3 = num_task_sessions*output_size
                
            input_ph = tf.placeholder(tf.float32, shape=[None, N1])
            target_ph = tf.placeholder(tf.float32, shape=[None, N3])

            W = tf.get_variable('Wi', shape=[N1, num_hidden], initializer=var_scale_init)
            b = tf.get_variable('Bi', shape=[num_hidden,], initializer=tf.zeros_initializer)
            hidden = nonlinearity(tf.matmul(input_ph, W) + b)

            for layer_i in range(1, num_layers-1):
                W = tf.get_variable('Wh%i' % layer_i, shape=[num_hidden, num_hidden], initializer=var_scale_init)
                b = tf.get_variable('B%i' % layer_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
                hidden = nonlinearity(tf.matmul(hidden, W) + b)

            W = tf.get_variable('Wo', shape=[num_hidden, N3], initializer=var_scale_init)
            b = tf.get_variable('Bo', shape=[N3,], initializer=tf.zeros_initializer)
            output = nonlinearity(tf.matmul(hidden, W) + b)
            
            domain_losses = []
            for task_i in range(num_task_sessions):
                if continue_training_all:
                    domain_losses.append(tf.nn.l2_loss(output[:, :(task_i+1)*output_size] - target_ph[:, :(task_i+1)*output_size]))
                else:
                    domain_losses.append(tf.nn.l2_loss(output[:, task_i*output_size:(task_i+1)*output_size] - target_ph[:, task_i*output_size:(task_i+1)*output_size]))


            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            domain_trains = []
            for task_i in range(num_task_sessions):
                domain_trains.append(optimizer.minimize(domain_losses[task_i]))	
        
            with tf.Session() as sess:
                def train_epoch(task_i):
                    if continue_training_all:
                        this_x_data = x_data[:x_l*(task_i+1), :]
                        this_y_data = y_data[:y_l*(task_i+1), :]
                    else:
                        this_x_data = x_data[x_l*task_i:x_l*(task_i+1), :]
                        this_y_data = y_data[y_l*task_i:y_l*(task_i+1), :]
                    this_order = np.random.permutation(len(this_x_data))
                    for batch_i in xrange(len(this_order)//batch_size):
                        sess.run(domain_trains[task_i], feed_dict={input_ph: this_x_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], target_ph: this_y_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]})


                def evaluate():
                    losses = []
                    for task_i in range(num_task_sessions): 
                        this_x_data = x_data[x_l*task_i:x_l*(task_i+1), :]
                        this_y_data = y_data[y_l*task_i:y_l*(task_i+1), :]
                        losses.append(sess.run(domain_losses[task_i], feed_dict={input_ph: this_x_data, target_ph: this_y_data}))
                    return [l/x_l for l in losses]
                
                sess.run(tf.global_variables_initializer())
                    
                with open("%s%s.csv" % (output_dir, filename_prefix), "w") as fout:
                    fout.write("epoch, " + ", ".join(["loss%i" % i for i in range(num_task_sessions)]) + "\n")
                    format_string = "%i, " + ", ".join(["%f" for _ in range(num_task_sessions)]) + "\n"
                    losses = evaluate()
                    curr_output = format_string % tuple([0] + losses)
                    print(curr_output)
                    fout.write(curr_output)
                    for task_i in xrange(num_task_sessions):
                        for epoch_i in xrange(task_i*num_epochs+1, (task_i+1)*num_epochs + 1):
                            train_epoch(task_i)	
                            if epoch_i % save_every == 0:
                                losses = evaluate()
                                curr_output = format_string % tuple([epoch_i] + losses)
                                print(curr_output)
                                fout.write(curr_output)
                                if losses[task_i] < early_stopping_thresh:
                                    print("Early stop task %i!" % task_i)
                                    break


            tf.reset_default_graph()
