from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from scipy.spatial.distance import pdist
import datasets

PI = np.pi
### Parameters
num_input = 4
num_output = 1
num_hidden = 16
num_hidden_hyper = 64
num_runs = 20 
learning_rate = 0.000033
meta_learning_rate = 0.000033

max_base_epochs = 1000 
max_meta_epochs = 500 
max_new_epochs = 500 
#num_layers = 3
output_dir = "meta_results/"
save_every = 10 #20
save_every_meta = 10
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
train_sequentially = True # If true, train task 2 and then task 1

batch_size = 32
meta_batch_size = 12 # how much of each dataset you see each meta train step
early_stopping_thresh = 0.005
meta_early_stopping_thresh = 0.01
base_tasks = ["X0", "NOTX0", "X0NOTX1", "XOR",  "OR", "AND"]
base_task_repeats = 6 # how many times each base task is seen
new_tasks = ["AND", "X0", "X0NOTX1", "XOR", "NXOR", "XOR_of_XORs"]
###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

internal_nonlinearity = tf.nn.leaky_relu
if tf_pm:
    output_nonlinearity = tf.nn.tanh
else:
    output_nonlinearity = tf.nn.sigmoid 


def _get_dataset(task, num_input):
    if task == "X0":
        x_data, y_data = datasets.X0_dataset(num_input)
    elif task == "NOTX0":
        x_data, y_data = datasets.NOTX0_dataset(num_input)
    elif task == "X0NOTX1":
        x_data, y_data = datasets.X0NOTX1_dataset(num_input)
    elif task == "XOR":
        x_data, y_data = datasets.XOR_dataset(num_input)
    elif task == "XOR_of_XORs":
        x_data, y_data = datasets.XOR_of_XORs_dataset(num_input)
    elif task == "NXOR":
        x_data, y_data = datasets.NXOR_dataset(num_input)
    elif task == "AND":
        x_data, y_data = datasets.AND_dataset(num_input)
    elif task == "OR":
        x_data, y_data = datasets.OR_dataset(num_input)
    elif task == "parity":
        x_data, y_data = datasets.parity_dataset(num_input)

    if tf_pm:
        x_data = 2*x_data - 1
        y_data = 2*y_data - 1

    perm = np.random.permutation(num_input)
    x0 = np.where(perm == 0)[0][0] # index of X0 in permuted data
    x1 = np.where(perm == 1)[0][0]

    x_data = x_data[:, perm] # shuffle columns so not always same inputs matter
    return {"x": x_data, "y": y_data, "relevant": [x0, x1]}

class meta_model(object):
    """A meta-learning model for binary functions."""
    def __init__(self, num_input, base_tasks, base_task_repeats, new_tasks):
        self.num_input = num_input
        self.num_output = num_output

        # base datasets
        self.base_tasks = []
        nbdpp = 2**num_input
        nbdpt = len(base_tasks) * base_task_repeats * nbdpp
        self.base_x_data = np.zeros([nbdpt, num_input]) 
        self.base_y_data = np.zeros([nbdpt, num_output]) 
        self.base_task_indices = np.zeros([nbdpt]) 

        for task_i, task_name in enumerate(base_tasks * base_task_repeats):
            dataset = _get_dataset(task_name, num_input)
            self.base_x_data[task_i*nbdpp:(task_i+1)*nbdpp, :] = dataset["x"]
            self.base_y_data[task_i*nbdpp:(task_i+1)*nbdpp, :] = dataset["y"]
            self.base_task_indices[task_i*nbdpp:(task_i+1)*nbdpp] = task_i
            self.base_tasks.append(task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";old" + str(task_i)) 

        # new datasets
        self.new_tasks = []
        self.new_datasets = {}
        for task_name in new_tasks:
            dataset = _get_dataset(task_name, num_input)
            task_full_name = task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";new"
            self.new_datasets[task_full_name] = dataset
            self.new_tasks.append(task_full_name) 

        self.num_tasks = num_tasks = len(self.base_tasks) + len(self.new_tasks)
        self.task_to_index = dict(zip(self.base_tasks + self.new_tasks, range(num_tasks)))


        # network
        self.base_input_ph = tf.placeholder(tf.float32, shape=[None, num_input])
        self.base_target_ph = tf.placeholder(tf.float32, shape=[None, num_output])
        self.task_index_ph = tf.placeholder(tf.int64, shape=[None])

        # hyper_network 
        self.function_embeddings = tf.get_variable('function_embedding', shape=[num_tasks, num_hidden_hyper],
                                                  initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))

        self.assign_f_emb_ph = tf.placeholder(tf.float32, shape=[num_hidden_hyper]) 
        self.assign_f_ind_ph = tf.placeholder(tf.int64, shape=[1]) 
        self.assign_f_emb = tf.scatter_update(self.function_embeddings, self.assign_f_ind_ph, tf.expand_dims(self.assign_f_emb_ph, axis=0))

        function_embedding = tf.nn.embedding_lookup(self.function_embeddings, self.task_index_ph)


        hyper_hidden_1 = slim.fully_connected(function_embedding, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity)
        hyper_hidden_2 = slim.fully_connected(hyper_hidden_1, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity)

        task_weights = slim.fully_connected(hyper_hidden_2, num_hidden*(num_input + num_hidden + num_output),
                                            activation_fn=None)
        task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_input + num_hidden + num_output)]) 
        
        task_biases = slim.fully_connected(hyper_hidden_2, num_hidden + num_hidden + num_output,
                                           activation_fn=None)

        # task network
        W1 = tf.transpose(task_weights[:, :, :num_input], perm=[0, 2, 1])
        W2 = task_weights[:, :,  num_input:num_input + num_hidden]
        W3 = task_weights[:, :,  -num_output:]
        b1 = task_biases[:, :num_hidden]
        b2 = task_biases[:, num_hidden:2*num_hidden]
        b3 = task_biases[:, 2*num_hidden:2*num_hidden + num_output]


        task_hidden_1 = internal_nonlinearity(tf.matmul(tf.expand_dims(self.base_input_ph, 1), W1) + tf.expand_dims(b1, 1))
        task_hidden_2 = internal_nonlinearity(tf.matmul(task_hidden_1, W2) + tf.expand_dims(b2, 1))
        self.output = tf.squeeze(output_nonlinearity(tf.matmul(task_hidden_2, W3) + tf.expand_dims(b3, 1)), axis=1)
        
        self.base_loss = tf.reduce_sum(tf.square(self.output - self.base_target_ph), axis=1)
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        base_full_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.base_full_train = base_full_optimizer.minimize(self.base_loss)
        
        base_emb_optimizer = tf.train.AdamOptimizer(learning_rate) # optimizes only embedding
        self.base_emb_train = base_emb_optimizer.minimize(self.base_loss,
                                                          var_list=[v for v in tf.global_variables() if "function_embedding" in v.name])

        # function "guessing" network 
        self.guess_input_ph = tf.placeholder(tf.float32, shape=[None, num_input + num_output])
        self.guess_target_ph = tf.placeholder(tf.float32, shape=[num_hidden_hyper])

        guess_hidden_1 = slim.fully_connected(self.guess_input_ph, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2 = slim.fully_connected(guess_hidden_1, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2b = tf.reduce_max(guess_hidden_2, axis=0, keep_dims=True)

        self.guess_output = slim.fully_connected(guess_hidden_2b, num_hidden_hyper,
                                                 activation_fn=None)

        self.guess_loss =  tf.nn.l2_loss(self.guess_output - self.guess_target_ph) 
        guess_optimizer = tf.train.AdamOptimizer(meta_learning_rate)
        self.guess_train = guess_optimizer.minimize(self.guess_loss)

        # initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        

    def base_eval(self):
        """Evaluates loss on the base tasks."""
        losses = np.zeros([len(self.base_tasks)])
        counts = np.zeros([len(self.base_tasks)])
        base_task_numbers = range(len(self.base_tasks))
        order = range(len(self.base_y_data))
        for batch_i in range(len(self.base_y_data)//batch_size):
            indices = order[batch_i*batch_size:(batch_i+1) * batch_size]
            this_batch_task_indices = self.base_task_indices[indices]
            this_feed_dict = {
                self.base_input_ph: self.base_x_data[indices, :],
                self.base_target_ph: self.base_y_data[indices, :],
                self.task_index_ph: this_batch_task_indices 
            }
            this_losses = self.sess.run(self.base_loss, feed_dict=this_feed_dict)
            for task_i in base_task_numbers: 
                this_task_indices = this_batch_task_indices == task_i
                losses[task_i] += np.sum(this_losses[this_task_indices])
                counts[task_i] += np.sum(this_task_indices)

        return losses/counts


    def new_eval(self, new_task):
        """Evaluates loss on a new task."""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        index = self.task_to_index[new_task_full_name]
        tiled_index = np.tile(index, len(dataset["x"])) 
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.base_target_ph: dataset["y"],
            self.task_index_ph: tiled_index 
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        count = len(dataset["y"])

        return loss/count


    def train_base_tasks(self, filename):
        """Train model to perform base tasks."""
        with open(filename, "w") as fout:
            fout.write("epoch, " + ", ".join(self.base_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks]) + "\n"

            for epoch in range(max_base_epochs):
                order = np.random.permutation(len(self.base_task_indices))
                for batch_i in range(len(order)//batch_size):
                    indices = order[batch_i*batch_size:(batch_i+1) * batch_size]
                    this_feed_dict = {
                        self.base_input_ph: self.base_x_data[indices, :],
                        self.base_target_ph: self.base_y_data[indices, :],
                        self.task_index_ph: self.base_task_indices[indices]
                    }
                    self.sess.run(self.base_full_train, feed_dict=this_feed_dict)

                if epoch % save_every == 0:
                    curr_losses = self.base_eval()
                    curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
                    fout.write(curr_output)
                    print(curr_output)
                    if np.all(curr_losses < early_stopping_thresh):
                        print("Early stop base!")
                        break


    def train_meta_task(self, filename):
        """Trains network to predict embeddings of the base tasks from the task
           data."""
        with open(filename, "w") as fout:
            fout.write("epoch, " + ", ".join(self.base_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks]) + "\n"
            print("Training meta")
            base_task_embeddings = self.sess.run(self.function_embeddings) 
#            print(pdist(base_task_embeddings))

            def _dataset_to_meta_point(task_i):
                this_task_indices = self.base_task_indices == task_i
                return np.concatenate([self.base_x_data[this_task_indices, :],
                                       self.base_y_data[this_task_indices, :]*self.num_input/self.num_output],
                                      axis=1), base_task_embeddings[task_i, :] 

            meta_sets = [_dataset_to_meta_point(task_i) for task_i in range(len(self.base_tasks))]
            curr_losses = np.zeros(len(meta_sets))

            for epoch in range(max_meta_epochs): 
                meta_order = np.random.permutation(len(meta_sets))
                for task_index in meta_order:
                    data, embedding = meta_sets[task_index] 
                    order = np.random.permutation(len(data))
                    order = order[:meta_batch_size]
                    this_feed_dict = {
                        self.guess_input_ph: data[order, :],
                        self.guess_target_ph: embedding
                    }
                    _, this_loss = self.sess.run([self.guess_train, self.guess_loss], feed_dict=this_feed_dict)
                    curr_losses[task_index] = this_loss

                curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
                fout.write(curr_output)
                print(curr_output)

                if np.all(curr_losses < meta_early_stopping_thresh):
                    print("Stopping meta early!")
                    break


    def train_new_tasks(self, filename_prefix):
        for new_task in new_tasks:
            print("Now training new task: " + new_task)
            new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
            print(new_task_full_name)
            print(self.task_to_index)
            index = self.task_to_index[new_task_full_name]
            dataset = self.new_datasets[new_task_full_name]

            # meta-network guess at optimal embedding
            this_feed_dict = {
                self.guess_input_ph: np.concatenate([dataset["x"],
                                                     dataset["y"]*self.num_input/self.num_output],
                                                    axis=1)
            }
            embedding_guess = self.sess.run(self.guess_output,
                                            feed_dict=this_feed_dict)[0] 

            with open(filename_prefix + new_task + ".csv", "w") as fout:
                fout.write("epoch, loss\n")
                curr_loss = self.new_eval(new_task) # random embedding 
                curr_output = "%i, %f\n" % (-1, curr_loss) 
                fout.write(curr_output)
                print(curr_output)

                # update with meta network guess
                self.sess.run(self.assign_f_emb, feed_dict={
                        self.assign_f_emb_ph: embedding_guess,
                        self.assign_f_ind_ph: np.array([index])
                    })

                curr_loss = self.new_eval(new_task) # guess embedding 
                curr_output = "%i, %f\n" % (0, curr_loss) 
                fout.write(curr_output)
                print(curr_output)

                # now tune only embedding
                for epoch in range(1, max_new_epochs):
                    order = np.random.permutation(len(dataset["x"]))
                    for batch_i in range(len(order)//batch_size + 1):
                        indices = order[batch_i*batch_size:(batch_i+1) * batch_size]
                        this_y = dataset["y"][indices, :]
                        tiled_index = np.tile(index, len(this_y)) 
                        this_feed_dict = {
                            self.base_input_ph: dataset["x"][indices, :],
                            self.base_target_ph: this_y,
                            self.task_index_ph: tiled_index
                        }
                        self.sess.run(self.base_emb_train, feed_dict=this_feed_dict)

                    if epoch % save_every == 0:
                        curr_loss = self.new_eval(new_task) # guess embedding 
                        curr_output = "%i, %f\n" % (epoch, curr_loss) 
                        fout.write(curr_output)
                        print(curr_output)
                        if curr_loss < early_stopping_thresh:
                            print("Early stop new!")
                            break

    def save_embeddings(self, filename):
        """Saves all task embeddings"""
        with open(filename, "w") as fout:
            fout.write("dimension, " + ", ".join(self.base_tasks + self.new_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks + self.new_tasks]) + "\n"
            task_embeddings = self.sess.run(self.function_embeddings) 
            for i in range(num_hidden_hyper):
                fout.write(("%i, " %i) + (format_string % tuple(task_embeddings[:, i])))
                

                
## running stuff

for run_i in xrange(num_runs):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    filename_prefix = "run%i" %(run_i)
    print("Now running %s" % filename_prefix)

    model = meta_model(num_input, base_tasks, base_task_repeats, new_tasks) 
    model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
    model.train_meta_task(filename=output_dir + filename_prefix + "_meta_losses.csv")
    model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
    model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")


    tf.reset_default_graph()

