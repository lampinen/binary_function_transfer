from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
PI = np.pi
### Parameters
num_input_per = 5
num_hidden = 32
num_runs = 10 
learning_rate = 0.0005
num_epochs = 20000
num_layers = 4
output_dir = "results_5/"
save_every = 20
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
train_sequentially = True # If true, train task 2 and then task 1
batch_size = 4
early_stopping_thresh = 0.005

###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')
if tf_pm:
    nonlinearity = tf.nn.tanh
else:
    nonlinearity = tf.nn.sigmoid 


for run_i in xrange(num_runs):
    for input_shared in [False, True]:
        for t1 in ["parity", "XOR_of_XORs", "XOR", "AND"]:
            for t2 in [ "X0", "XOR", "XOR_of_XORs", "parity", "OR", "AND", "None"]:
                np.random.seed(run_i)
                tf.set_random_seed(run_i)
                filename_prefix = "t1%s_t2%s_sharedinput%s_run%i" %(t1, t2, str(input_shared), run_i)
                print("Now running %s" % filename_prefix)
                if t1 == "X0":
                    x1_data, y1_data = datasets.X0_dataset(num_input_per)
                elif t1 == "XOR":
                    x1_data, y1_data = datasets.XOR_dataset(num_input_per)
                elif t1 == "XOR_of_XORs":
                    x1_data, y1_data = datasets.XOR_of_XORs_dataset(num_input_per)
                elif t1 == "NXOR":
                    x1_data, y1_data = datasets.NXOR_dataset(num_input_per)
                elif t1 == "AND":
                    x1_data, y1_data = datasets.AND_dataset(num_input_per)
                elif t1 == "OR":
                    x1_data, y1_data = datasets.OR_dataset(num_input_per)
                elif t1 == "parity":
                    x1_data, y1_data = datasets.parity_dataset(num_input_per)

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
                elif t2 == "OR":
                    x2_data, y2_data = datasets.OR_dataset(num_input_per)
                elif t2 == "parity":
                    x2_data, y2_data = datasets.parity_dataset(num_input_per)

                if t2 == "None":
                    if input_shared:
                        x_data = x1_data
                    else:
                        x_data = np.concatenate([x1_data, np.zeros_like(x1_data)], axis=1)
                    y_data = np.concatenate([y1_data, np.zeros_like(y1_data)], axis=1)
                else:
                    if input_shared:
                        x_data = np.concatenate([x1_data, x2_data], axis=0)
                    else:
                        x_data = np.concatenate([np.concatenate([x1_data, np.zeros_like(x2_data)], axis=1),
                                                 np.concatenate([np.zeros_like(x1_data), x2_data], axis=1)], axis=0)
                    y_data = np.concatenate([np.concatenate([y1_data, np.zeros_like(y2_data)], axis=1),
                                             np.concatenate([np.zeros_like(y1_data), y2_data], axis=1)], axis=0)

                if tf_pm:
                    x_data = 2*x_data - 1
                    y_data = 2*y_data - 1

                output_size = 1
    
                num_datapoints = len(x_data)
                batch_subset = len(x1_data)
                
                if input_shared:
                    N1 = num_input_per
                else:
                    N1 = 2*num_input_per
                    
                input_ph = tf.placeholder(tf.float32, shape=[None, N1])
                target_ph = tf.placeholder(tf.float32, shape=[None, 2*output_size])

                W = tf.get_variable('Wi', shape=[N1, num_hidden], initializer=var_scale_init)
                b = tf.get_variable('Bi', shape=[num_hidden,], initializer=tf.zeros_initializer)
                hidden = nonlinearity(tf.matmul(input_ph, W) + b)

                for layer_i in range(1, num_layers-1):
                    W = tf.get_variable('Wh%i' % layer_i, shape=[num_hidden, num_hidden], initializer=var_scale_init)
                    b = tf.get_variable('B%i' % layer_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
                    hidden = nonlinearity(tf.matmul(hidden, W) + b)

                W = tf.get_variable('Wo', shape=[num_hidden, 2*output_size], initializer=var_scale_init)
                b = tf.get_variable('Bo', shape=[2*output_size,], initializer=tf.zeros_initializer)
                output = nonlinearity(tf.matmul(hidden, W) + b)
                

                domain_one_mask = np.concatenate([np.ones(batch_subset, dtype=np.bool), np.zeros(batch_subset, dtype=np.bool)], axis=0)
                d1_mask_ph = tf.placeholder(tf.bool, shape=[None,])
                first_domain_loss = tf.nn.l2_loss(tf.boolean_mask(output[:, :output_size] - target_ph[:, :output_size], d1_mask_ph))
                second_domain_loss = tf.nn.l2_loss(tf.boolean_mask(output[:, output_size:] - target_ph[:, output_size:], tf.logical_not(d1_mask_ph)))

                if t2 == "None":
                    loss = first_domain_loss
                else:
                    loss = first_domain_loss + second_domain_loss
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                train = optimizer.minimize(loss)	
                fd_train = optimizer.minimize(first_domain_loss)
                sd_train = optimizer.minimize(second_domain_loss)
            
                with tf.Session() as sess:
                    def train_epoch():
                        this_order = np.random.permutation(num_datapoints)
                        for batch_i in xrange(num_datapoints//batch_size):
                            sess.run(train, feed_dict={input_ph: x_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], target_ph: y_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], d1_mask_ph: domain_one_mask[this_order[batch_i*batch_size:(batch_i+1)*batch_size]]})

                    def train_epoch_1():
                        this_order = np.random.permutation(num_datapoints)
                        for batch_i in xrange(num_datapoints//batch_size):
                            sess.run(fd_train, feed_dict={input_ph: x_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], target_ph: y_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], d1_mask_ph: domain_one_mask[this_order[batch_i*batch_size:(batch_i+1)*batch_size]]})

                    def train_epoch_2():
                        this_order = np.random.permutation(num_datapoints)
                        for batch_i in xrange(num_datapoints//batch_size):
                            sess.run(sd_train, feed_dict={input_ph: x_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], target_ph: y_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], d1_mask_ph: domain_one_mask[this_order[batch_i*batch_size:(batch_i+1)*batch_size]]})


                    def evaluate():
                        curr_loss1 = 0.
                        curr_loss2 = 0.
                        for batch_i in xrange(num_datapoints//batch_size):
                            curr_loss1 += sess.run(first_domain_loss, feed_dict={input_ph: x_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y_data[batch_i*batch_size:(batch_i+1)*batch_size, :], d1_mask_ph: domain_one_mask[batch_i*batch_size:(batch_i+1)*batch_size]})
                            curr_loss2 += sess.run(second_domain_loss, feed_dict={input_ph: x_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y_data[batch_i*batch_size:(batch_i+1)*batch_size, :], d1_mask_ph: domain_one_mask[batch_i*batch_size:(batch_i+1)*batch_size]})
                        return curr_loss1/batch_subset, curr_loss2/batch_subset
                    
                    sess.run(tf.global_variables_initializer())
                        
                    with open("%s%s.csv" % (output_dir, filename_prefix), "w") as fout:
                        fout.write("epoch, loss1, loss2\n")
                        loss1, loss2 = evaluate()
                        print("%i, %f, %f\n" % (0, loss1, loss2))
                        fout.write("%i, %f, %f\n" % (0, loss1, loss2))
                        if train_sequentially:
                            if t2 != "None":
                                for epoch_i in xrange(1, num_epochs + 1):
                                    train_epoch_2()	
                                    if epoch_i % save_every == 0:
                                        loss1, loss2 = evaluate()
                                        print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                        fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                        if loss2 < early_stopping_thresh:
                                            print("Early stop prior!")
                                            break
                            for epoch_i in xrange(num_epochs+1, 2*num_epochs + 1):
                                train_epoch() # train on both	
                                if epoch_i % save_every == 0:
                                    loss1, loss2 = evaluate()
                                    print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    if loss1 < early_stopping_thresh:
                                        print("Early stop main!")
                                        break

                        else:
                            for epoch_i in xrange(1, num_epochs + 1):
                                train_epoch()	
                                if epoch_i % save_every == 0:
                                    loss1, loss2 = evaluate()
                                    print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    if loss1 < early_stopping_thresh and loss2 < early_stopping_thresh:
                                        print("Early stop!")
                                        break

                tf.reset_default_graph()
