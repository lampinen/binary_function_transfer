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
learning_rate = 3e-5
meta_learning_rate = 1e-5 

max_base_epochs = 4000 
max_meta_epochs = 2000 
max_new_epochs = 500 
num_task_hidden_layers = 3
num_meta_hidden_layers = 3
output_dir = "meta_results/"
save_every = 10 #20
save_every_meta = 10
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
hyper_convolutional = True # whether hyper network creates weights convolutionally
conv_in_channels = 4

batch_size = 16
meta_batch_size = 12 # how much of each dataset the function embedding guesser sees 
early_stopping_thresh = 0.005
meta_early_stopping_thresh = 0.001
base_tasks = ["X0", "NOTX0", "XOR", "NOR", "OR", "AND"]
base_task_repeats = 5 # how many times each base task is seen
new_tasks = ["X0", "AND", "OR", "NOR", "NAND", "X0NOTX1", "XOR", "XOR_of_XORs"]
###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

internal_nonlinearity = tf.nn.leaky_relu
if tf_pm:
    output_nonlinearity = tf.nn.tanh
else:
    output_nonlinearity = tf.nn.sigmoid 

# TODO: update for general inputs
perm_list_template = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1), (2, 0, 1, 3), (2, 0, 3, 1), (2, 3, 0, 1)]
single_perm_list_template = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
total_tasks = set(base_tasks + new_tasks)
perm_list_dict = {task: (np.random.permutation(perm_list_template) if task not in ["XO", "NOTX0", "threeparity"] else np.random.permutation(single_perm_list_template)) for task in total_tasks} 

def _get_perm(task):
    perm = np.copy(perm_list_dict[task][0, :])
    perm_list_dict[task] = np.delete(perm_list_dict[task], 0, 0)
    return perm

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
    elif task == "NAND":
        x_data, y_data = datasets.NAND_dataset(num_input)
    elif task == "NOR":
        x_data, y_data = datasets.NOR_dataset(num_input)
    elif task == "parity":
        x_data, y_data = datasets.parity_dataset(num_input)
    elif task == "threeparity":
        x_data, y_data = datasets.parity_dataset(num_input, num_to_keep=3)

    if tf_pm:
        x_data = 2*x_data - 1
        y_data = 2*y_data - 1

    perm = _get_perm(task) 
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
        self.base_datasets = {}

        for task_i, task_name in enumerate(base_tasks * base_task_repeats):
            dataset = _get_dataset(task_name, num_input)
            task_full_name = task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";old" + str(task_i)
            self.base_datasets[task_full_name] = dataset 
            self.base_tasks.append(task_full_name) 

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

        # function embedding "guessing" network 
        self.guess_input_ph = tf.placeholder(tf.float32, shape=[None, num_input + num_output])

        guess_hidden_1 = slim.fully_connected(self.guess_input_ph, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2 = slim.fully_connected(guess_hidden_1, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2b = tf.reduce_max(guess_hidden_2, axis=0, keep_dims=True)

        self.function_embedding = slim.fully_connected(guess_hidden_2b, num_hidden_hyper,
                                                       activation_fn=None)

        # hyper_network 

        hyper_hidden = self.function_embedding
        for _ in range(num_meta_hidden_layers):
            hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                activation_fn=internal_nonlinearity)
        
        self.hidden_weights = []
        self.hidden_biases = []
        if hyper_convolutional: 
            hyper_hidden = slim.fully_connected(hyper_hidden, conv_in_channels*(num_task_hidden_layers*num_hidden + num_output),
                                                  activation_fn=internal_nonlinearity)
            hyper_hidden = tf.reshape(hyper_hidden, [-1, (num_task_hidden_layers*num_hidden + num_output), conv_in_channels])



            Wi = slim.convolution(hyper_hidden[:, :num_hidden, :], num_input,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            Wi = tf.transpose(Wi, perm=[0, 2, 1])
            bi = slim.convolution(hyper_hidden[:, :num_hidden, :], 1,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            bi = tf.squeeze(bi, axis=-1)
            self.hidden_weights.append(Wi)
            self.hidden_biases.append(bi)

            for i in range(1, num_task_hidden_layers):
                Wi = slim.convolution(hyper_hidden[:, num_hidden:2*num_hidden, :], num_hidden,
                                      kernel_size=1, padding='SAME',
                                      activation_fn=None)
                Wi = tf.transpose(Wi, perm=[0, 2, 1])
                self.hidden_weights.append(Wi)
                bi = slim.convolution(hyper_hidden[:, num_hidden:2*num_hidden, :], 1,
                                      kernel_size=1, padding='SAME',
                                      activation_fn=None)
                bi = tf.squeeze(bi, axis=-1)
                self.hidden_biases.append(bi)

            Wfinal = slim.convolution(hyper_hidden[:, -num_output:, :], num_hidden,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            Wfinal = tf.transpose(Wfinal, perm=[0, 2, 1])
            bfinal = slim.convolution(hyper_hidden[:, -num_output:, :], 1,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            bfinal = tf.squeeze(bfinal, axis=-1)

        else:
            hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                  activation_fn=internal_nonlinearity)
            task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_input + num_hidden + num_output),
                                                activation_fn=None)

            task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_input + num_hidden + num_output)]) 

            task_biases = slim.fully_connected(hyper_hidden, num_hidden + num_hidden + num_output,
                                               activation_fn=None)

            W1 = tf.transpose(task_weights[:, :, :num_input], perm=[0, 2, 1])
            W2 = task_weights[:, :,  num_input:num_input + num_hidden]
            W3 = task_weights[:, :,  -num_output:]
            b1 = task_biases[:, :num_hidden]
            b2 = task_biases[:, num_hidden:2*num_hidden]
            b3 = task_biases[:, 2*num_hidden:2*num_hidden + num_output]


        for i in range(num_task_hidden_layers):
            self.hidden_weights[i] = tf.squeeze(self.hidden_weights[i], axis=0)
            self.hidden_biases[i] = tf.squeeze(self.hidden_biases[i], axis=0)

        Wfinal = tf.squeeze(Wfinal, axis=0)
        bfinal = tf.squeeze(bfinal, axis=0)

        # task network
        task_hidden = self.base_input_ph
        for i in range(num_task_hidden_layers):
            task_hidden = internal_nonlinearity(tf.matmul(task_hidden, self.hidden_weights[i]) + self.hidden_biases[i])
        self.output = output_nonlinearity(tf.matmul(task_hidden, Wfinal) + bfinal)
        
        self.base_loss = tf.reduce_sum(tf.square(self.output - self.base_target_ph), axis=1)
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        base_full_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.base_full_train = base_full_optimizer.minimize(self.total_base_loss)

        # initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    
    def dataset_eval(self, dataset):
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: dataset["x"],
            self.base_target_ph: dataset["y"]
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        return loss


    def dataset_trainstep(self, dataset):
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: dataset["x"],
            self.base_target_ph: dataset["y"]
        }
        loss = self.sess.run(self.base_full_train, feed_dict=this_feed_dict)
        return loss


    def base_eval(self):
        """Evaluates loss on the base tasks."""
        losses = np.zeros([len(self.base_tasks)])
        for task in self.base_tasks:
            dataset =  self.base_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        return losses


    def new_eval(self, new_task):
        """Evaluates loss on a new task."""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        return self.dataset_eval(dataset)


    def new_outputs(self, new_task):
        """Returns outputs on a new task."""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        index = self.task_to_index[new_task_full_name]
        tiled_index = np.tile(index, len(dataset["x"])) 
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: dataset["x"]
        }
        outputs = self.sess.run(self.output, feed_dict=this_feed_dict)
        return outputs


    def train_base_tasks(self, filename_prefix):
        """Train model to perform base tasks as meta task."""
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
                with open(filename_prefix + new_task + "_outputs.csv", "w") as foutputs:
                    foutputs.write("type, " + ', '.join(["input%i" % i for i in range(len(dataset["y"]))]) + "\n")
                    foutputs.write("target, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(dataset["y"].flatten()) + "\n")

                    fout.write("epoch, %s\n" % new_task_full_name)
                    curr_loss = self.new_eval(new_task) # random embedding 
                    curr_net_outputs = self.new_outputs(new_task)
                    foutputs.write("random_emb, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")
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
                    curr_net_outputs = self.new_outputs(new_task)
                    foutputs.write("guess_emb, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")

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
                    curr_net_outputs = self.new_outputs(new_task)
                    foutputs.write("trained_emb, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")


    def save_embeddings(self, filename):
        """Saves all task embeddings"""
        def _simplify(t):
            split_t = t.split(';')
            return ';'.join([split_t[0], split_t[2]])
        with open(filename, "w") as fout:
            simplified_tasks = [_simplify(t) for t in self.base_tasks + self.new_tasks]
            fout.write("dimension, " + ", ".join(simplified_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks + self.new_tasks]) + "\n"
            task_embeddings = self.sess.run(self.function_embeddings) 
            for i in range(num_hidden_hyper):
                fout.write(("%i, " %i) + (format_string % tuple(task_embeddings[:, i])))
                

                
## running stuff

for run_i in xrange(num_runs):
    np.random.seed(run_i)
    perm_list_dict = {task: (np.random.permutation(perm_list_template) if task not in ["XO", "NOTX0", "threeparity"] else np.random.permutation(single_perm_list_template)) for task in total_tasks} 
    tf.set_random_seed(run_i)
    filename_prefix = "run%i" %(run_i)
    print("Now running %s" % filename_prefix)

    model = meta_model(num_input, base_tasks, base_task_repeats, new_tasks) 
    model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
    model.train_meta_task(filename=output_dir + filename_prefix + "_meta_losses.csv")
    model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
    model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")


    tf.reset_default_graph()

