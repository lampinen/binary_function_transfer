from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
import os
PI = np.pi
### Parameters
num_input_per = 5
num_hidden = 50
num_runs = 200 
num_test = 4 # how many datapoints to hold out for eval 
learning_rate = 0.01
num_epochs = 40000
num_layers = 5
init_mult = 0.33 
output_dir = "results_generalization_complex_stb_nh_%i_lr_%.4f_im_%.2f/" %(num_hidden, learning_rate, init_mult)
save_every = 5
train_sequentially = True # If true, train task 2 and then task 1
second_train_both = True # If train_sequentially, whether to continue training on task 2 while training task 1
batch_size = 4
early_stopping_thresh = 5e-4
###
output_size = num_input_per
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=init_mult, mode='FAN_AVG')
nonlinearity = tf.nn.leaky_relu
output_nonlinearity = tf.nn.sigmoid 


for input_shared in [False]:#, True]:
    for run_i in range(num_runs):
#        for t1 in ["XOR_of_XORs", "XOR", "XAO", "ANDXORS", "AND"]:
#            for t2 in ["X0", "XOR", "XOR_of_XORs", "XAO", "OR", "ANDXORS", "AND", "None"]:
        for t1 in ["IDENTITY", "LOO_PARITY", "PAIR_AND", "PAIR_XOR", "MIX1", "MIX2"]:
            for t2 in ["IDENTITY", "LOO_PARITY", "PAIR_AND", "PAIR_XOR", "MIX1", "MIX2", "None"]:
                np.random.seed(run_i)
                tf.set_random_seed(run_i)
                filename_prefix = "t1%s_t2%s_sharedinput%s_run%i" %(t1, t2, str(input_shared), run_i)
                print("Now running %s" % filename_prefix)
#                if t1 == "X0":
#                    x1_data, y1_data = datasets.X0_dataset(num_input_per)
#                elif t1 == "XOR":
#                    x1_data, y1_data = datasets.XOR_dataset(num_input_per)
#                elif t1 == "XOR_of_XORs":
#                    x1_data, y1_data = datasets.XOR_of_XORs_dataset(num_input_per)
#                elif t1 == "NXOR":
#                    x1_data, y1_data = datasets.NXOR_dataset(num_input_per)
#                elif t1 == "AND":
#                    x1_data, y1_data = datasets.AND_dataset(num_input_per)
#                elif t1 == "OR":
#                    x1_data, y1_data = datasets.OR_dataset(num_input_per)
#                elif t1 == "parity":
#                    x1_data, y1_data = datasets.parity_dataset(num_input_per)
#                elif t1 == "XAO":
#                    x1_data, y1_data = datasets.XAO_dataset(num_input_per)
#                elif t1 == "ANDXORS":
#                    x1_data, y1_data = datasets.ANDXORS_dataset(num_input_per)
#
#                if t2 == "X0":
#                    x2_data, y2_data = datasets.X0_dataset(num_input_per)
#                elif t2 == "XOR":
#                    x2_data, y2_data = datasets.XOR_dataset(num_input_per)
#                elif t2 == "XOR_of_XORs":
#                    x2_data, y2_data = datasets.XOR_of_XORs_dataset(num_input_per)
#                elif t2 == "NXOR":
#                    x2_data, y2_data = datasets.NXOR_dataset(num_input_per)
#                elif t2 == "AND":
#                    x2_data, y2_data = datasets.AND_dataset(num_input_per)
#                elif t2 == "OR":
#                    x2_data, y2_data = datasets.OR_dataset(num_input_per)
#                elif t2 == "parity":
#                    x2_data, y2_data = datasets.parity_dataset(num_input_per)
#                elif t2 == "XAO":
#                    x2_data, y2_data = datasets.XAO_dataset(num_input_per)
#                elif t2 == "ANDXORS":
#                    x2_data, y2_data = datasets.ANDXORS_dataset(num_input_per)
                if t1 == "IDENTITY":
                    x1_data, y1_data = datasets.IDENTITY_dataset(num_input_per)
                elif t1 == "LOO_PARITY":
                    x1_data, y1_data = datasets.LOO_PARITY_dataset(num_input_per)
                elif t1 == "PAIR_AND":
                    x1_data, y1_data = datasets.PAIR_AND_dataset(num_input_per)
                elif t1 == "PAIR_OR":
                    x1_data, y1_data = datasets.AND_dataset(num_input_per)
                elif t1 == "MIX1":
                    x1_data, y1_data = datasets.OR_dataset(num_input_per)
                elif t1 == "MIX2":
                    x1_data, y1_data = datasets.MIX2_dataset(num_input_per)

                if t2 == "IDENTITY":
                    x2_data, y2_data = datasets.IDENTITY_dataset(num_input_per)
                elif t2 == "LOO_PARITY":
                    x2_data, y2_data = datasets.LOO_PARITY_dataset(num_input_per)
                elif t2 == "PAIR_AND":
                    x2_data, y2_data = datasets.PAIR_AND_dataset(num_input_per)
                elif t2 == "PAIR_OR":
                    x2_data, y2_data = datasets.AND_dataset(num_input_per)
                elif t2 == "MIX1":
                    x2_data, y2_data = datasets.OR_dataset(num_input_per)
                elif t2 == "MIX2":
                    x2_data, y2_data = datasets.MIX2_dataset(num_input_per)

                num_datapoints = len(x1_data)
                num_train = num_datapoints - num_test

                order1 = np.random.permutation(num_datapoints)
                order2 = np.random.permutation(num_datapoints)
                x1_test_data = x1_data[order1[-num_test:]]
                y1_test_data = y1_data[order1[-num_test:]]
                x1_data = x1_data[order1[:num_train]]
                y1_data = y1_data[order1[:num_train]]

                if t2 != "None":
                    x2_test_data = x2_data[order2[-num_test:]]
                    y2_test_data = y2_data[order2[-num_test:]]
                    x2_data = x2_data[order2[:num_train]]
                    y2_data = y2_data[order2[:num_train]]

                input_1_ph = tf.placeholder(tf.float32, shape=[None, num_input_per])
                input_2_ph = tf.placeholder(tf.float32, shape=[None, num_input_per])
                target_ph = tf.placeholder(tf.float32, shape=[None, output_size])

                W1 = tf.get_variable('Widom1', shape=[num_input_per, num_hidden], initializer=var_scale_init)
                b1 = tf.get_variable('Bidom1', shape=[num_hidden,], initializer=tf.zeros_initializer)
                W2 = tf.get_variable('Widom2', shape=[num_input_per, num_hidden], initializer=var_scale_init)
                b2 = tf.get_variable('Bidom2', shape=[num_hidden,], initializer=tf.zeros_initializer)
                hidden1 = nonlinearity(tf.matmul(input_1_ph, W1) + b1)
                hidden2 = nonlinearity(tf.matmul(input_2_ph, W2) + b2)

                for layer_i in range(1, num_layers-1):
                    W = tf.get_variable('Wh%i' % layer_i, shape=[num_hidden, num_hidden], initializer=var_scale_init)
                    b = tf.get_variable('B%i' % layer_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
                    hidden1 = nonlinearity(tf.matmul(hidden1, W) + b)
                    hidden2 = nonlinearity(tf.matmul(hidden2, W) + b)

                W1 = tf.get_variable('Wodom1', shape=[num_hidden, output_size], initializer=var_scale_init)
                b1 = tf.get_variable('Bodom1', shape=[output_size,], initializer=tf.zeros_initializer)
                W2 = tf.get_variable('Wodom2', shape=[num_hidden, output_size], initializer=var_scale_init)
                b2 = tf.get_variable('Bodom2', shape=[output_size,], initializer=tf.zeros_initializer)
                logits1 = tf.matmul(hidden1, W1) + b1
                logits2 = tf.matmul(hidden2, W2) + b2
                output1 = output_nonlinearity(logits1)
                output2 = output_nonlinearity(logits2)

                first_domain_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits1, labels=target_ph)
                first_domain_loss = tf.reduce_mean(first_domain_loss)
                second_domain_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits2, labels=target_ph) 
                second_domain_loss = tf.reduce_mean(second_domain_loss)
                    
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                fd_train = optimizer.minimize(first_domain_loss)
                sd_train = optimizer.minimize(second_domain_loss)
            
                sess_config = tf.ConfigProto()
                sess_config.gpu_options.allow_growth = True

                with tf.Session(config=sess_config) as sess:
                    def train_epoch_1():
                        this_order = np.random.permutation(num_train)
                        for batch_i in range(num_train//batch_size):
                            feed_dict = {input_1_ph: x1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]} 
                            sess.run(fd_train, feed_dict=feed_dict)

                    def train_epoch_2():
                        this_order = np.random.permutation(num_train)
                        for batch_i in range(num_train//batch_size):
                            feed_dict = {input_2_ph: x2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                         target_ph: y2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                            sess.run(sd_train, feed_dict=feed_dict)

                    def train_epoch():
                        this_order = np.random.permutation(num_train)
                        for batch_i in range(num_train//batch_size):
                            feed_dict = {input_1_ph: x1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]} 
                            sess.run(fd_train, feed_dict=feed_dict)
                            feed_dict = {input_2_ph: x2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                         target_ph: y2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                            sess.run(sd_train, feed_dict=feed_dict)


                    def evaluate():
                        curr_loss1 = 0.
                        curr_loss2 = 0.
                        for batch_i in range(num_train//batch_size):
                            curr_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        if t2 != "None":
                            for batch_i in range(num_train//batch_size):
                                curr_loss2 += sess.run(second_domain_loss, feed_dict={input_2_ph: x2_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y2_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        curr_test_loss1 = 0.
                        curr_test_loss2 = 0.
                        for batch_i in range(num_test//batch_size):
                            curr_test_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        if t2 != "None":
                            for batch_i in range(num_test//batch_size):
                                curr_test_loss2 += sess.run(second_domain_loss, feed_dict={input_2_ph: x2_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y2_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        return curr_loss1/num_train, curr_test_loss1/num_test, curr_loss2/num_train, curr_test_loss2/num_test
                    
                    sess.run(tf.global_variables_initializer())
                        
                    with open("%s%s.csv" % (output_dir, filename_prefix), "w") as fout:
                        fout.write("epoch, loss1, loss1_test, loss2, loss2_test\n")
                        losses = evaluate()
                        this_result = "%i, %f, %f, %f, %f\n" % tuple([0] + [l for l in losses])
                        print(this_result)
                        fout.write(this_result)
                        if train_sequentially:
                            if t2 != "None":
                                for epoch_i in range(1, num_epochs + 1):
                                    train_epoch_2()        
                                    if epoch_i % save_every == 0:
                                        losses = evaluate()
                                        this_result = "%i, %f, %f, %f, %f\n" % tuple([epoch_i] + [l for l in losses])
                                        print(this_result)
                                        fout.write(this_result)
                                        if losses[2] < early_stopping_thresh:
                                            print("Early stop prior!")
                                            break
                            for epoch_i in range(num_epochs+1, 2*num_epochs + 1):
                                if second_train_both: 
                                    train_epoch() # train on both        
                                else: 
                                    train_epoch_1()
                                if epoch_i % save_every == 0:
                                    losses = evaluate()
                                    this_result = "%i, %f, %f, %f, %f\n" % tuple([epoch_i] + [l for l in losses])
                                    print(this_result)
                                    fout.write(this_result)
                                    if losses[0] < early_stopping_thresh:
                                        print("Early stop main!")
                                        break

                        else:
                            for epoch_i in range(1, num_epochs + 1):
                                train_epoch()        
                                if epoch_i % save_every == 0:
                                    losses = evaluate()
                                    this_result = "%i, %f, %f, %f, %f\n" % tuple([epoch_i] + [l for l in losses])
                                    print(this_result)
                                    fout.write(this_result)
                                    if losses[0] < early_stopping_thresh:
                                        print("Early stop!")
                                        break

                tf.reset_default_graph()
