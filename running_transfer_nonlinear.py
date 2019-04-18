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
num_hidden = 10
num_runs = 50 
learning_rate = 0.005
num_epochs = 20000
num_layers = 4
init_mult = 1. 
output_dir = "results_synaptic_intelligence_nh_%i_lr_%.4f_im_%.2f/" %(num_hidden, learning_rate, init_mult)
save_every = 5
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
#train_sequentially = True # If true, train task 2 and then task 1
#second_train_both = False # If train_sequentially, whether to continue training on task 2 while training task 1
batch_size = 4
early_stopping_thresh = 0.005

# parameters for the synaptic intelligence 
synaptic_intelligence_weight = 1e-1
stability_xi = 1e-1
###
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=init_mult, mode='FAN_AVG')
if tf_pm:
    nonlinearity = tf.nn.tanh
else:
    nonlinearity = tf.nn.sigmoid 


for input_shared in [False]:#, True]:
    for run_i in xrange(num_runs):
        for t1 in ["XOR_of_XORs", "XOR", "XAO", "AND"]:
            for t2 in ["None", "X0", "XOR", "XOR_of_XORs", "XAO", "OR", "AND"]:
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
                elif t1 == "XAO":
                    x1_data, y1_data = datasets.XAO_dataset(num_input_per)

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
                elif t2 == "XAO":
                    x2_data, y2_data = datasets.XAO_dataset(num_input_per)

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
                output1 = nonlinearity(tf.matmul(hidden1, W1) + b1)
                output2 = nonlinearity(tf.matmul(hidden2, W2) + b2)

                first_domain_loss = tf.nn.l2_loss(output1 - target_ph)
                second_domain_loss = tf.nn.l2_loss(output2 - target_ph)

                trainable_vars = tf.trainable_variables()
                # N.B. it would be better to do these computations in the graph
                # instead of passing them in, but this implementation is easier 

                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                sd_train = optimizer.minimize(second_domain_loss)

                sd_grads_and_vars = [(g, v) for (g,v) in optimizer.compute_gradients(second_domain_loss) if g is not None]
                sd_trainable_vars = [v for g,v in sd_grads_and_vars]

                reference_w_placeholders = [tf.placeholder(tf.float32, shape=var.get_shape()) for var in sd_trainable_vars]
                reg_strength_placeholders = [tf.placeholder(tf.float32, shape=var.get_shape()) for var in sd_trainable_vars]
                
                parameter_wise_regs = [tf.reduce_sum(tf.multiply(reg_strength_placeholders[i], tf.square(v-reference_w_placeholders[i]))) for i, v in enumerate(sd_trainable_vars)] 

                surrogate_loss = synaptic_intelligence_weight * tf.add_n(parameter_wise_regs) 

                if t2 == "None":
                    fd_train = optimizer.minimize(first_domain_loss)
                else:
                    fd_train = optimizer.minimize(first_domain_loss + surrogate_loss)

                sess_config = tf.ConfigProto()
                sess_config.gpu_options.allow_growth = True
            
                with tf.Session(config=sess_config) as sess:
#                    def train_epoch():
#                        this_order = np.random.permutation(num_datapoints)
#                        for batch_i in xrange(num_datapoints//batch_size):
#                            sess.run(train, feed_dict={input_ph: x_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], target_ph: y_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], d1_mask_ph: domain_one_mask[this_order[batch_i*batch_size:(batch_i+1)*batch_size]]})
#
                    def train_epoch_1(reg_strengths=None, reference_ws=None):
                        apply_int_syn = reg_strengths is not None
                        this_order = np.random.permutation(batch_subset)
                        for batch_i in xrange(batch_subset//batch_size):
                            feed_dict = {input_1_ph: x1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                         target_ph: y1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                            if apply_int_syn: # won't be applied if other task is None 
                                for i in range(len(sd_trainable_vars)):
                                    feed_dict[reference_w_placeholders[i]] = reference_ws[i] 
                                    feed_dict[reg_strength_placeholders[i]] = reg_strengths[i] 
                            sess.run(fd_train, feed_dict=feed_dict)

                    def train_epoch_2():
                        this_order = np.random.permutation(batch_subset)
                        for batch_i in xrange(batch_subset//batch_size):
                            feed_dict = {input_2_ph: x2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                         target_ph: y2_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                            old_weights_and_grads = sess.run(sd_grads_and_vars, feed_dict=feed_dict)
                            sess.run(sd_train, feed_dict=feed_dict)
                            new_weights = sess.run(sd_trainable_vars)
                            for var_i in range(len(new_weights)):
                                grad, old_weight = old_weights_and_grads[var_i]
                                reg_strengths[var_i] -= grad * (new_weights[var_i] - old_weight) 


                    if t2 == "None":
                        def evaluate():
                            curr_loss1 = 0.
                            for batch_i in xrange(batch_subset//batch_size):
                                curr_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                            return curr_loss1/batch_subset, 0. 

                    else:
                        def evaluate():
                            curr_loss1 = 0.
                            curr_loss2 = 0.
                            for batch_i in xrange(batch_subset//batch_size):
                                curr_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                                curr_loss2 += sess.run(second_domain_loss, feed_dict={input_2_ph: x2_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y2_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                            return curr_loss1/batch_subset, curr_loss2/batch_subset
                    
                    sess.run(tf.global_variables_initializer())
                        
                    with open("%s%s.csv" % (output_dir, filename_prefix), "w") as fout:
                        fout.write("epoch, loss1, loss2\n")
                        loss1, loss2 = evaluate()
                        print("%i, %f, %f\n" % (0, loss1, loss2))
                        fout.write("%i, %f, %f\n" % (0, loss1, loss2))
                        if t2 != "None":
                            initial_weight_vals = sess.run(sd_trainable_vars)
                            reg_strengths = [np.zeros_like(v) for v in initial_weight_vals]
                            for epoch_i in xrange(1, num_epochs + 1):
                                train_epoch_2()	
                                if epoch_i % save_every == 0:
                                    loss1, loss2 = evaluate()
                                    print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                    if loss2 < early_stopping_thresh:
                                        print("Early stop prior!")
                                        break
                            reference_weight_vals = sess.run(sd_trainable_vars)
                            for var_i in range(len(reference_weight_vals)):
                                reg_strengths[var_i] /= np.square(reference_weight_vals[var_i] - initial_weight_vals[var_i]) + stability_xi 

                                print(sd_trainable_vars[var_i])
                                print(reg_strengths[var_i])
                        for epoch_i in xrange(num_epochs+1, 2*num_epochs + 1):
#                                if second_train_both: 
#                                    train_epoch() # train on both	
#                                else: 
                            if t2 != "None":
                                train_epoch_1(reg_strengths=reg_strengths,
                                              reference_ws=reference_weight_vals)
                            else:
                                train_epoch_1()
                            if epoch_i % save_every == 0:
                                loss1, loss2 = evaluate()
                                print("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                fout.write("%i, %f, %f\n" % (epoch_i, loss1, loss2))
                                if loss1 < early_stopping_thresh:
                                    print("Early stop main!")
                                    break


                tf.reset_default_graph()
