from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
import os
PI = np.pi
### Parameters
num_input = 50
num_output = 50
num_source_examples = 4400
num_target_examples = 800
ground_truth_bottleneck = 5 
num_prior_tasks = 1
num_hidden = 50
num_runs = 500 
num_test = 400 # how many datapoints to hold out for eval 
learning_rate = 1e-4
num_epochs = 10000
num_layers = 5
ground_truth_hidden_layers = 2 #num_layers - 1
init_mult = 0.033
optimizer_name = "Adam"
output_dir = "results_moretask_generalization_nonbinary_scales_%s_stb_nt2_%i_gthl_%i_gtb_%i_nse_%i_nte_%i_nl_%i_nh_%i_lr_%.4f_im_%.2f/" %(optimizer_name, num_prior_tasks, ground_truth_hidden_layers, ground_truth_bottleneck, num_source_examples, num_target_examples, num_layers, num_hidden, learning_rate, init_mult)
save_every = 5
train_sequentially = True # If true, train task 2 and then task 1
second_train_both = True # If train_sequentially, whether to continue training on task 2 while training task 1
batch_size = 400
early_stopping_thresh = 1e-4
###
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=init_mult, mode='FAN_AVG')
nonlinearity = tf.nn.leaky_relu
output_nonlinearity = nonlinearity


for input_shared in [False]:#, True]:
    for run_i in range(num_runs):
        for t1 in [0, 1]:
            for t2 in ["None", 0, 1]:
                np.random.seed(run_i)
                tf.set_random_seed(run_i)
                filename_prefix = "t1%s_t2%s_sharedinput%s_run%i" %(str(t1), str(t2), str(input_shared), run_i)
                print("Now running %s" % filename_prefix)
                x1_data, y1_data = datasets.random_low_rank_function(
                    num_input, num_output, num_target_examples, seed=2 * run_i + t1,
                    num_hidden_layers=ground_truth_hidden_layers,
                    rank=ground_truth_bottleneck)

                num_datapoints_1 = len(x1_data)
                num_train_1 = num_datapoints_1 - num_test

                order1 = np.random.permutation(num_datapoints_1)
                x1_test_data = x1_data[order1[-num_test:]]
                y1_test_data = y1_data[order1[-num_test:]]
                x1_data = x1_data[order1[:num_train_1]]
                y1_data = y1_data[order1[:num_train_1]]

                if t2 != "None":
                    x2_datas = []
                    y2_datas = []
                    x2_test_datas = []
                    y2_test_datas = []
                    for t2_i in range(num_prior_tasks):
                        x2_data, y2_data = datasets.random_low_rank_function(
                            num_input, num_output, num_source_examples, seed=2 * run_i + t2,
                        num_hidden_layers=ground_truth_hidden_layers,
                        rank=ground_truth_bottleneck)
                        num_datapoints_2 = len(x2_data)
                        num_train_2 = num_datapoints_2 - num_test
                        order2 = np.random.permutation(num_datapoints_2)
                        x2_test_data = x2_data[order2[-num_test:]]
                        y2_test_data = y2_data[order2[-num_test:]]
                        x2_data = x2_data[order2[:num_train_2]]
                        y2_data = y2_data[order2[:num_train_2]]
                        x2_datas.append(x2_data)
                        y2_datas.append(y2_data)
                        x2_test_datas.append(x2_test_data)
                        y2_test_datas.append(y2_test_data)
                else:
                    num_train_2 = 1  # for a dummy division

                input_1_ph = tf.placeholder(tf.float32, shape=[None, num_input])

                input_2_phs = []
                for t2_i in range(num_prior_tasks):
                    input_2_ph = tf.placeholder(tf.float32, shape=[None, num_input])
                    input_2_phs.append(input_2_ph)
                target_ph = tf.placeholder(tf.float32, shape=[None, num_output])

                W1 = tf.get_variable('Widom1', shape=[num_input, num_hidden], initializer=var_scale_init)
                b1 = tf.get_variable('Bidom1', shape=[num_hidden,], initializer=tf.zeros_initializer)
                hidden1 = nonlinearity(tf.matmul(input_1_ph, W1) + b1)

                hidden2s = []
                for t2_i in range(num_prior_tasks):
                    W2 = tf.get_variable('Widom2%i' % t2_i, shape=[num_input, num_hidden], initializer=var_scale_init)
                    b2 = tf.get_variable('Bidom2%i' % t2_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
                    hidden2 = nonlinearity(tf.matmul(input_2_phs[t2_i], W2) + b2)
                    hidden2s.append(hidden2)

                for layer_i in range(1, num_layers-1):
                    W = tf.get_variable('Wh%i' % layer_i, shape=[num_hidden, num_hidden], initializer=var_scale_init)
                    b = tf.get_variable('B%i' % layer_i, shape=[num_hidden,], initializer=tf.zeros_initializer)
                    hidden1 = nonlinearity(tf.matmul(hidden1, W) + b)
                    for t2_i in range(num_prior_tasks):
                        hidden2s[t2_i] = nonlinearity(tf.matmul(hidden2s[t2_i], W) + b)

                W1 = tf.get_variable('Wodom1', shape=[num_hidden, num_output], initializer=var_scale_init)
                b1 = tf.get_variable('Bodom1', shape=[num_output,], initializer=tf.zeros_initializer)
                logits1 = tf.matmul(hidden1, W1) + b1
                output1 = output_nonlinearity(logits1)

                logits2s = []
                output2s = []
                for t2_i in range(num_prior_tasks):
                    W2 = tf.get_variable('Wodom2%i' % t2_i, shape=[num_hidden, num_output], initializer=var_scale_init)
                    b2 = tf.get_variable('Bodom2%i' % t2_i, shape=[num_output,], initializer=tf.zeros_initializer)
                    logits2 = tf.matmul(hidden2s[t2_i], W2) + b2
                    output2 = output_nonlinearity(logits2)
                    logits2s.append(logits2)
                    output2s.append(output2)

                first_domain_loss = tf.nn.l2_loss(output1 - target_ph)
                first_domain_loss = tf.reduce_mean(first_domain_loss)

                second_domain_losses = []

                for t2_i in range(num_prior_tasks):
                    second_domain_loss = tf.nn.l2_loss(output2s[t2_i] - target_ph) 
                    second_domain_loss = tf.reduce_mean(second_domain_loss)
                    second_domain_losses.append(second_domain_loss)
                    
                if optimizer_name == "Adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                elif optimizer_name == "sgd":
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                else:
                    raise ValueError("Unknown optimizer!")
                fd_train = optimizer.minimize(first_domain_loss)

                sd_trains = []

                for t2_i in range(num_prior_tasks):
                    sd_train = optimizer.minimize(second_domain_losses[t2_i])
                    sd_trains.append(sd_train)
            
                sess_config = tf.ConfigProto()
                sess_config.gpu_options.allow_growth = True

                with tf.Session(config=sess_config) as sess:
                    def train_epoch_1():
                        this_order = np.random.permutation(num_train_1)
                        for batch_i in range(num_train_1//batch_size):
                            feed_dict = {input_1_ph: x1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]} 
                            sess.run(fd_train, feed_dict=feed_dict)

                    def train_epoch_2():
                        this_order = np.random.permutation(num_train_2)
                        for batch_i in range(num_train_2//batch_size):
                            for t2_i in range(num_prior_tasks):
                                feed_dict = {input_2_phs[t2_i]: x2_datas[t2_i][this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y2_datas[t2_i][this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                                sess.run(sd_trains[t2_i], feed_dict=feed_dict)

                    def train_epoch():
                        this_order = np.random.permutation(num_train_1)
                        this_order_2 = np.random.permutation(num_train_2)
                        for batch_i in range(num_train_1//batch_size):  # note that we only replay a subset of domain 2
                            feed_dict = {input_1_ph: x1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y1_data[this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]} 
                            sess.run(fd_train, feed_dict=feed_dict)
                            for t2_i in range(num_prior_tasks):
                                feed_dict = {input_2_phs[t2_i]: x2_datas[t2_i][this_order[batch_i*batch_size:(batch_i+1)*batch_size], :], 
                                             target_ph: y2_datas[t2_i][this_order[batch_i*batch_size:(batch_i+1)*batch_size], :]}
                                sess.run(sd_trains[t2_i], feed_dict=feed_dict)


                    def evaluate():
                        curr_loss1 = 0.
                        curr_loss2s = [0.] * num_prior_tasks
                        for batch_i in range(num_train_1//batch_size):
                            curr_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        if t2 != "None":
                            for t2_i in range(num_prior_tasks):
                                for batch_i in range(num_train_2//batch_size):
                                    curr_loss2s[t2_i] += sess.run(second_domain_losses[t2_i], feed_dict={input_2_phs[t2_i]: x2_datas[t2_i][batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y2_datas[t2_i][batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        curr_test_loss1 = 0.
                        curr_test_loss2s = [0.] * num_prior_tasks
                        for batch_i in range(num_test//batch_size):
                            curr_test_loss1 += sess.run(first_domain_loss, feed_dict={input_1_ph: x1_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y1_test_data[batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        if t2 != "None":
                            for t2_i in range(num_prior_tasks):
                                for batch_i in range(num_test//batch_size):
                                    curr_test_loss2s[t2_i] += sess.run(second_domain_losses[t2_i], feed_dict={input_2_phs[t2_i]: x2_test_datas[t2_i][batch_i*batch_size:(batch_i+1)*batch_size, :], target_ph: y2_test_datas[t2_i][batch_i*batch_size:(batch_i+1)*batch_size, :]})
                        t2_losses = zip([l / num_train_2 for l in curr_loss2s],
                                        [l / num_test for l in curr_test_loss2s])
                        t2_losses = [l for l2 in t2_losses for l in l2]
                        return [curr_loss1/num_train_1, curr_test_loss1/num_test] + t2_losses 
                    
                    sess.run(tf.global_variables_initializer())
                        
                    with open("%s%s.csv" % (output_dir, filename_prefix), "w") as fout:
                        fout.write("epoch, loss1, loss1_test, " + ", ".join(["loss2%i, loss2%i_test" for t2_i in range(num_prior_tasks)]) + "\n")
                        losses = evaluate()
                        format_string = "%i," + ", ".join(["%f, %f"] * (num_prior_tasks + 1)) + "\n"
                        this_result = format_string % tuple([0] + [l for l in losses])
                        print(this_result)
                        fout.write(this_result)
                        if train_sequentially:
                            if t2 != "None":
                                for epoch_i in range(1, num_epochs + 1):
                                    train_epoch_2()        
                                    if epoch_i % save_every == 0:
                                        losses = evaluate()
                                        this_result = format_string % tuple([epoch_i] + [l for l in losses])
                                        print(this_result)
                                        fout.write(this_result)
                                        if np.all(np.array(losses[2::2]) < early_stopping_thresh):
                                            print("Early stop prior!")
                                            break
                            for epoch_i in range(num_epochs+1, 2*num_epochs + 1):
                                if second_train_both and t2 != "None": 
                                    train_epoch() # train on both        
                                else: 
                                    train_epoch_1()
                                if epoch_i % save_every == 0:
                                    losses = evaluate()
                                    this_result = format_string % tuple([epoch_i] + [l for l in losses])
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
                                    this_result = format_string % tuple([epoch_i] + [l for l in losses])
                                    print(this_result)
                                    fout.write(this_result)
                                    if losses[0] < early_stopping_thresh:
                                        print("Early stop!")
                                        break

                tf.reset_default_graph()
