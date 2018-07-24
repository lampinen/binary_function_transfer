from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from scipy.spatial.distance import pdist
import datasets

pi = np.pi
### Parameters
num_input = 4
num_output = 1
num_hidden = 64
num_hidden_hyper = 64
num_runs = 20 
init_learning_rate = 1e-4
new_init_learning_rate = 1e-7
lr_decay = 0.8
lr_decays_every = 100
min_learning_rate = 1e-6
refresh_meta_cache_every = 1#200 # how many epochs between updates to meta_dataset_cache

train_momentum = 0.8
adam_epsilon = 1e-3

max_base_epochs = 5000 
max_new_epochs = 1000 
num_task_hidden_layers = 3
num_meta_hidden_layers = 3
output_dir = "meta_results/"
save_every = 10 #20
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
hyper_convolutional = False # whether hyper network creates weights convolutionally
conv_in_channels = 6

batch_size = 16
meta_batch_size = 12 # how much of each dataset the function embedding guesser sees 
early_stopping_thresh = 0.005
base_tasks = ["X0", "NOTX0", "X0NOTX1", "NOTX0NOTX1", "OR", "AND", "NOTAND"]
base_meta_tasks = ["ID", "NOT"]
base_task_repeats = 5 # how many times each base task is seen
new_tasks = ["X0", "AND", "OR", "NOTOR", "NOTAND", "XOR", "XOR_of_XORs"]
###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

internal_nonlinearity = tf.nn.leaky_relu
if tf_pm:
    output_nonlinearity = tf.nn.tanh
else:
    output_nonlinearity = tf.nn.sigmoid 

# TODO: update for general inputs
perm_list_template = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1), (2, 0, 1, 3), (2, 0, 3, 1), (2, 3, 0, 1)]
#perm_list_template = [(0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 2, 3, 1, 4), (2, 0, 1, 3, 4), (2, 0, 3, 1, 4), (2, 3, 0, 1, 4)]
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
    elif task == "NOTX0NOTX1":
        x_data, y_data = datasets.NOTX0NOTX1_dataset(num_input)
    elif task == "XOR":
        x_data, y_data = datasets.XOR_dataset(num_input)
    elif task == "XOR_of_XORs":
        x_data, y_data = datasets.XOR_of_XORs_dataset(num_input)
    elif task == "NOTXOR":
        x_data, y_data = datasets.NXOR_dataset(num_input)
    elif task == "AND":
        x_data, y_data = datasets.AND_dataset(num_input)
    elif task == "OR":
        x_data, y_data = datasets.OR_dataset(num_input)
    elif task == "NOTAND":
        x_data, y_data = datasets.NAND_dataset(num_input)
    elif task == "NOTOR":
        x_data, y_data = datasets.NOR_dataset(num_input)
    elif task == "parity":
        x_data, y_data = datasets.parity_dataset(num_input)
    elif task == "threeparity":
        x_data, y_data = datasets.parity_dataset(num_input, num_to_keep=3)

    if tf_pm:
        x_data = 2*x_data - 1
        y_data = 2*y_data - 1

    # shuffle columns so not always same inputs matter
    perm = _get_perm(task) 
    x0 = np.where(perm == 0)[0][0] # index of X0 in permuted data
    x1 = np.where(perm == 1)[0][0]

    x_data = x_data[:, perm] 

    #padding
    x_data = np.concatenate([x_data, np.zeros([len(x_data), num_hidden_hyper - num_input])], axis=1)
    y_data = np.concatenate([np.zeros([len(y_data), num_hidden_hyper - num_output]), y_data], axis=1)
    return {"x": x_data, "y": y_data, "relevant": [x0, x1]}

class meta_model(object):
    """A meta-learning model for binary functions."""
    def __init__(self, num_input, base_tasks, base_task_repeats, new_tasks):
        self.num_input = num_input
        self.num_output = num_output

        # base datasets
        self.base_tasks = []
        self.base_datasets = {}

        for task_i, task_name in enumerate(base_tasks * base_task_repeats):
            dataset = _get_dataset(task_name, num_input)
            task_full_name = task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";old" + str(task_i)
            self.base_datasets[task_full_name] = dataset 
            self.base_tasks.append(task_full_name) 

        self.base_meta_tasks = base_meta_tasks 
        self.meta_dataset_cache = {t: {} for t in base_meta_tasks} # will cache datasets for a certain number of epochs
                                                                   # both to speed training and to keep training targets
                                                                   # consistent

        # new datasets
        self.new_tasks = []
        self.new_datasets = {}
        for task_name in new_tasks:
            dataset = _get_dataset(task_name, num_input)
            task_full_name = task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";new"
            self.new_datasets[task_full_name] = dataset
            self.new_tasks.append(task_full_name) 

        self.all_tasks = self.base_tasks + self.new_tasks + self.base_meta_tasks
        self.num_tasks = num_tasks = len(self.all_tasks)
        self.task_to_index = dict(zip(self.all_tasks, range(num_tasks)))

        # network
        self.base_input_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.base_target_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.lr_ph = tf.placeholder(tf.float32)

        # function embedding "guessing" network 
        self.guess_input_ph = tf.placeholder(tf.float32, shape=[None, 2 * num_hidden_hyper])

        guess_hidden_1 = slim.fully_connected(self.guess_input_ph, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2 = slim.fully_connected(guess_hidden_1, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 
        guess_hidden_2b = tf.reduce_max(guess_hidden_2, axis=0, keep_dims=True)
        guess_hidden_3 = slim.fully_connected(guess_hidden_2b, num_hidden_hyper,
                                              activation_fn=internal_nonlinearity) 

        self.guess_function_embedding = slim.fully_connected(guess_hidden_3, num_hidden_hyper,
                                                             activation_fn=None)


        # hyper_network 
        self.embedding_is_fed = tf.placeholder_with_default(False, [])
        self.feed_embedding_ph = tf.placeholder_with_default(np.zeros([1, num_hidden_hyper], dtype=np.float32), shape=[1, num_hidden_hyper])
        self.function_embedding = tf.cond(self.embedding_is_fed, 
                                          lambda: self.feed_embedding_ph,
                                          lambda: self.guess_function_embedding)

        hyper_hidden = self.function_embedding
        for _ in range(num_meta_hidden_layers-1):
            hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                activation_fn=internal_nonlinearity)
        
        self.hidden_weights = []
        self.hidden_biases = []
        if hyper_convolutional: 
            hyper_hidden = slim.fully_connected(hyper_hidden, conv_in_channels*(num_task_hidden_layers*num_hidden + num_output),
                                                  activation_fn=internal_nonlinearity)
            hyper_hidden = tf.reshape(hyper_hidden, [-1, (num_task_hidden_layers*num_hidden + num_hidden_hyper), conv_in_channels])



            Wi = slim.convolution(hyper_hidden[:, :num_hidden, :], num_hidden_hyper,
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

            Wfinal = slim.convolution(hyper_hidden[:, -num_hidden_hyper:, :], num_hidden,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            Wfinal = tf.transpose(Wfinal, perm=[0, 2, 1])
            bfinal = slim.convolution(hyper_hidden[:, -num_hidden_hyper:, :], 1,
                                  kernel_size=1, padding='SAME',
                                  activation_fn=None)
            bfinal = tf.squeeze(bfinal, axis=-1)

        else:
            hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                  activation_fn=internal_nonlinearity)
            task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                activation_fn=None)

            task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)]) 
            task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                               activation_fn=None)

            Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
            bi = task_biases[:, :num_hidden]
            self.hidden_weights.append(Wi)
            self.hidden_biases.append(bi)
            for i in range(1, num_task_hidden_layers):
                Wi = tf.transpose(task_weights[:, :, num_input+(i-1)*num_hidden:num_input+i*num_hidden], perm=[0, 2, 1])
                bi = task_biases[:, num_hidden*i:num_hidden*(i+1)]
                self.hidden_weights.append(Wi)
                self.hidden_biases.append(bi)
            Wfinal = task_weights[:, :, -num_hidden_hyper:]
            bfinal = task_biases[:, -num_hidden_hyper:]

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
        

        self.base_is_masked = tf.placeholder_with_default(True, []) # whether to mask loss to just strict "output" units
        self.base_loss = tf.cond(self.base_is_masked,
            lambda: tf.reduce_sum(tf.square(self.output - self.base_target_ph)[:, -num_output:], axis=1),
            lambda: tf.reduce_sum(tf.square(self.output - self.base_target_ph), axis=1))
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        #base_full_optimizer = tf.train.MomentumOptimizer(self.lr_ph, train_momentum)
        base_full_optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        self.base_full_train = base_full_optimizer.minimize(self.total_base_loss)

        # initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    
    def _guess_dataset(self, dataset):
        return np.concatenate([dataset["x"], dataset["y"]],
                              axis=1)

    
    def dataset_eval(self, dataset, zeros=False, mask=True):
        guess_dataset = self._guess_dataset(dataset)
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: np.zeros_like(guess_dataset) if zeros else guess_dataset,
            self.base_is_masked: mask,
            self.base_target_ph: dataset["y"]
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        return loss


    def dataset_embedding_eval(self, dataset, embedding, zeros=False, mask=True):
        this_feed_dict = {
            self.embedding_is_fed: True,
            self.feed_embedding_ph: np.zeros_like(embedding) if zeros else embedding,
            self.guess_input_ph: np.zeros([1, 2*num_hidden_hyper]),
            self.base_input_ph: dataset["x"],
            self.base_is_masked: mask,
            self.base_target_ph: dataset["y"]
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        return loss


    def dataset_train_step(self, dataset, lr, mask=True):
        guess_subset = np.random.permutation(len(dataset["y"]))[:meta_batch_size]
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: self._guess_dataset(dataset)[guess_subset, :],
            self.base_target_ph: dataset["y"],
            self.base_is_masked: mask,
            self.lr_ph: lr
        }
        loss = self.sess.run(self.base_full_train, feed_dict=this_feed_dict)
        return loss
        

    def get_meta_dataset(self, meta_task):
        x_data = []
        y_data = []
        if meta_task == "NOT":
            for task in self.base_tasks: 
                stripped_task = ";".join(task.split(";")[:-1])
                other = "NOT" + stripped_task if task[:3] != "NOT" else stripped_task[3:]
                other_tasks = [t for t in self.base_tasks if ";".join(t.split(";")[:-1]) == other]
                if other_tasks != []:
                    other = other_tasks[0]
                    x_data.append(self.get_task_embedding(self.base_datasets[task])[0, :])
                    y_data.append(self.get_task_embedding(self.base_datasets[other])[0, :])
        elif meta_task == "ID":
            for task in self.base_tasks: 
                embedding = self.get_task_embedding(self.base_datasets[task])[0, :]
                x_data.append(embedding)
                y_data.append(embedding)

        return {"x": np.array(x_data), "y": np.array(y_data)}


    def meta_true_eval(self):
        # BROKEN
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        losses = []
        names = []
        tasks = self.base_tasks + self.new_tasks
        for meta_task in self.base_meta_tasks:
            meta_dataset = self.get_meta_dataset(meta_task)
            if meta_task == "NOT":
                for task in tasks:
		    stripped_task = ";".join(task.split(";")[:-1])
		    other = "NOT" + stripped_task if task[:3] != "NOT" else stripped_task[3:]
		    other_tasks = [t for t in tasks if ";".join(t.split(";")[:-1]) == other]
		    if other_tasks != []:
			other = other_tasks[0]
                       
                        if task in self.base_tasks:
                            task_dataset = self.base_datasets[task]
                        else: 
                            task_dataset = self.new_datasets[task]

                        task_embedding = self.get_task_embedding(task_dataset)

			if other in self.base_tasks:
			    dataset = self.base_datasets[other]
			else:
			    dataset = self.new_datasets[other]

                        mapped_embedding = self.get_outputs(meta_dataset,
                                                            {"x": task_embedding},
                                                            full=True)

                        names.append("NOT:" + task + "->" + other)
                        losses.append(self.dataset_embedding_eval(dataset, mapped_embedding))

            elif meta_task == "ID":
                for task in tasks:
                    if task in self.base_tasks:
                        dataset = self.base_datasets[task]
                    else: 
                        dataset = self.new_datasets[task]

                    task_embedding = self.get_task_embedding(dataset)

                    mapped_embedding = self.get_outputs(meta_dataset,
                                                        {"x": task_embedding},
                                                        full=True)

                    names.append("ID:" + task + "->" + task)
                    losses.append(self.dataset_embedding_eval(dataset, mapped_embedding))

            else:
                raise ValueError("Unrecognized meta task")

        return losses, names


    def base_eval(self):
        """Evaluates loss on the base tasks."""
        losses = np.zeros([len(self.base_tasks) + len(self.base_meta_tasks)])
        for task in self.base_tasks:
            dataset =  self.base_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        offset = len(self.new_tasks) # new come before meta in indices
        for task in self.base_meta_tasks:
            dataset = self.meta_dataset_cache[task]
            losses[self.task_to_index[task] - offset] = self.dataset_eval(dataset, mask=False)

        return losses


    def new_eval(self, new_task, zeros=False):
        """Evaluates loss on a new task."""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        return self.dataset_eval(dataset, zeros=zeros)


    def all_eval(self):
        """Evaluates loss on the base and new tasks."""
        losses = np.zeros([self.num_tasks])
        for task in self.base_tasks:
            dataset =  self.base_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        for task in self.base_meta_tasks:
            dataset = self.meta_dataset_cache[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset, mask=False)

        for task in self.new_tasks:
            dataset =  self.new_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        return losses

    def get_outputs(self, dataset, new_dataset=None, full=False):
        if new_dataset is None:
            new_dataset = guess_dataset
        this_feed_dict = {
            self.base_input_ph: new_dataset["x"],
            self.guess_input_ph: self._guess_dataset(dataset) 
        }
        outputs = self.sess.run(self.output, feed_dict=this_feed_dict)
        if full:
            return outputs
        else:
            return outputs[:, -1]


    def new_outputs(self, new_task, zeros=False):
        """Returns outputs on a new task.
           zeros: if True, will give empty dataset to guessing net, for
           baseline"""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        guess_dataset = self._guess_dataset(dataset)
        this_feed_dict = {
            self.base_input_ph: dataset["x"],
            self.guess_input_ph: np.zeros_like(guess_dataset) if zeros else guess_dataset,
        }
        outputs = self.sess.run(self.output, feed_dict=this_feed_dict)[:, -1]
        return outputs


    def train_base_tasks(self, filename):
        """Train model to perform base tasks as meta task."""
        with open(filename, "w") as fout:
            fout.write("epoch, " + ", ".join(self.base_tasks + self.base_meta_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks + self.base_meta_tasks]) + "\n"

            learning_rate = init_learning_rate

            for epoch in range(max_base_epochs):

                if epoch % refresh_meta_cache_every == 0:
                    for task in self.base_meta_tasks:
                        self.meta_dataset_cache[task] = self.get_meta_dataset(task)

                order = np.random.permutation(len(self.base_tasks))
                for task_i in order:
                    task = self.base_tasks[task_i]
                    dataset =  self.base_datasets[task]
                    self.dataset_train_step(dataset, learning_rate)

                order = np.random.permutation(len(self.base_meta_tasks))
                for task_i in order:
                    task = self.base_meta_tasks[task_i]
                    dataset =  self.meta_dataset_cache[task]
                    self.dataset_train_step(dataset, learning_rate, mask=False)


                if epoch % save_every == 0:
                    curr_losses = self.base_eval()
                    curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
                    fout.write(curr_output)
                    print(curr_output)
                    if np.all(curr_losses < early_stopping_thresh):
                        print("Early stop base!")
                        break

                if epoch % lr_decays_every == 0 and epoch > 0 and learning_rate > min_learning_rate:
                    learning_rate *= lr_decay


    def train_new_tasks(self, filename_prefix):
        print("Now training new tasks...")

        with open(filename_prefix + "new_losses.csv", "w") as fout:
            with open(filename_prefix + "meta_true_losses.csv", "w") as fout_meta:
                fout.write("epoch, " + ", ".join(self.base_tasks + self.new_tasks) + "\n")
                format_string = ", ".join(["%f" for _ in self.all_tasks]) + "\n"

                for new_task in self.new_tasks:
                    dataset = self.new_datasets[new_task]
                    with open(filename_prefix + new_task + "_outputs.csv", "w") as foutputs:
                        foutputs.write("type, " + ', '.join(["input%i" % i for i in range(len(dataset["y"]))]) + "\n")
                        foutputs.write("target, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(dataset["y"][:, -1].flatten()) + "\n")

                        curr_net_outputs = self.new_outputs(new_task, zeros=True)
                        foutputs.write("baseline, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")
                        curr_net_outputs = self.new_outputs(new_task, zeros=False)
                        foutputs.write("guess_emb, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")

                for task in self.base_meta_tasks:
                    self.meta_dataset_cache[task] = self.get_meta_dataset(task)

                curr_meta_true_losses, meta_true_names = self.meta_true_eval() 
                fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")
                meta_format_string = ", ".join(["%f" for _ in meta_true_names]) + "\n"
                curr_meta_output = ("0, ") + (meta_format_string % tuple(curr_meta_true_losses))
                fout_meta.write(curr_meta_output)

                curr_losses = self.all_eval() # guess embedding 
                print(len(curr_losses))
                curr_output = ("0, ") + (format_string % tuple(curr_losses))
                fout.write(curr_output)
                print(curr_output)

                # now tune
                learning_rate = new_init_learning_rate
                for epoch in range(1, max_new_epochs):
                    if epoch % refresh_meta_cache_every == 0:
                        for task in self.base_meta_tasks:
                            self.meta_dataset_cache[task] = self.get_meta_dataset(task)

                    order = np.random.permutation(self.num_tasks)
                    for task_i in order:
                        task = self.all_tasks[task_i]
                        mask=True
                        if task in self.new_tasks:
                            dataset =  self.new_datasets[task]
                        elif task in self.base_meta_tasks:
                            dataset = self.meta_dataset_cache[task]
                            mask=False
                        else:
                            dataset =  self.base_datasets[task]
                        self.dataset_train_step(dataset, learning_rate, 
                                                mask=mask)

                    if epoch % save_every == 0:
                        curr_meta_true_losses, _ = self.meta_true_eval() 
                        curr_meta_output = ("%i, " % epoch) + (meta_format_string % tuple(curr_meta_true_losses))
                        fout_meta.write(curr_meta_output)

                        curr_losses = self.all_eval()
                        curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
                        fout.write(curr_output)
                        print(curr_output)
                        if np.all(curr_losses < early_stopping_thresh):
                            print("Early stop new!")
                            break

                    if epoch % lr_decays_every == 0 and epoch > 0 and learning_rate > min_learning_rate:
                        learning_rate *= lr_decay

        for new_task in self.new_tasks:
            
            with open(filename_prefix + new_task + "_outputs.csv", "a") as foutputs:
                curr_net_outputs = self.new_outputs(new_task)
                foutputs.write("trained_emb, " + ', '.join(["%f" for i in range(len(curr_net_outputs))]) % tuple(curr_net_outputs) + "\n")


    def get_task_embedding(self, dataset):
        """Gets task embedding"""
        guess_dataset = self._guess_dataset(dataset)

        return self.sess.run(
            self.function_embedding,
            feed_dict={
                self.guess_input_ph: guess_dataset  
            }) 


    def save_embeddings(self, filename):
        """Saves all task embeddings"""
        def _simplify(t):
            split_t = t.split(';')
            return ';'.join([split_t[0], split_t[2]])
        with open(filename, "w") as fout:
            basic_tasks = self.base_tasks + self.new_tasks
            simplified_tasks = [_simplify(t) for t in basic_tasks]
            fout.write("dimension, " + ", ".join(simplified_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in basic_tasks]) + "\n"
            task_embeddings = np.zeros([len(basic_tasks), num_hidden_hyper])

            for task in basic_tasks:
                if task in self.new_tasks:
                    dataset =  self.new_datasets[task]
                else:
                    dataset =  self.base_datasets[task]
                task_i = self.task_to_index[task] 
                task_embeddings[task_i, :] = self.get_task_embedding(dataset)
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
    model.save_embeddings(filename=output_dir + filename_prefix + "_init_embeddings.csv")
    model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
    model.save_embeddings(filename=output_dir + filename_prefix + "_guess_embeddings.csv")
    model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
    model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")


    tf.reset_default_graph()

