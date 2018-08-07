from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from scipy.spatial.distance import pdist
import datasets
from orthogonal_matrices import random_orthogonal

pi = np.pi
### Parameters
num_input = 8
num_output = 1 # cannot be changed without somewhat substantial code modifications
num_hidden = 64
num_hidden_hyper = 64
num_runs = 20 
init_learning_rate = 3e-4
init_meta_learning_rate = 3e-4
new_init_learning_rate = 1e-6
new_init_meta_learning_rate = 1e-6
lr_decay = 0.8
meta_lr_decay = 0.8
lr_decays_every = 100
min_learning_rate = 5e-7
refresh_meta_cache_every = 1#200 # how many epochs between updates to meta_dataset_cache

train_momentum = 0.8
adam_epsilon = 1e-3

max_base_epochs = 3000 
max_new_epochs = 1000
num_task_hidden_layers = 3
num_meta_hidden_layers = 3
output_dir = "meta_results/"
save_every = 10 #20
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
cue_dimensions = True # if true, provide two-hot cues of which dimensions are relevant
hyper_convolutional = False # whether hyper network creates weights convolutionally
conv_in_channels = 6

batch_size = 256
meta_batch_size = 196 # how much of each dataset the function embedding guesser sees 
early_stopping_thresh = 0.005
base_tasks = ["X0", "NOTX0", "AND", "NOTAND", "X0NOTX1", "NOTX0NOTX1", "OR", "XOR", "NOTXOR"]
base_meta_tasks = ["ID", "NOT", "isX0", "isNOTX0", "isAND", "isNOTAND", "isXOR", "isNOTXOR"]
base_task_repeats = 27 # how many times each base task is seen
new_tasks = ["X0", "AND", "OR",  "X0NOTX1", "NOTX0NOTX1","NOTOR", "NOTAND", "XOR", "NOTXOR"]
###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

internal_nonlinearity = tf.nn.leaky_relu
if tf_pm:
    output_nonlinearity = tf.nn.tanh
else:
    output_nonlinearity = tf.nn.sigmoid 

# TODO: update for general inputs
def _get_perm_list_template(num_input):
    template = []
    for i in range(num_input):
        for j in range(i+1, num_input):
            this_perm = list(range(num_input)) 
            this_perm[j] = 1
            this_perm[1] = j
            temp = this_perm[i]
            this_perm[i] = 0
            this_perm[0] = temp 
            template.append(this_perm)
    return template


def _get_single_perm_list_template(num_input):
    template = []
    for offset in range(1, num_input):
        for i in range(num_input):
            j = i + offset
            if j >= num_input:
                break
            this_perm = list(range(num_input)) 
            this_perm[j] = 1
            this_perm[1] = j
            temp = this_perm[i]
            this_perm[i] = 0
            this_perm[0] = temp 
            template.append(this_perm)
    return template

total_tasks = set(base_tasks + new_tasks)
perm_list_dict = {task: (np.random.permutation(_get_perm_list_template(num_input)) if task not in ["XO", "NOTX0"] else np.random.permutation(_get_single_perm_list_template(num_input))) for task in total_tasks} 

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

    y_data = np.squeeze(y_data).astype(np.int32)

    if tf_pm:
        x_data = 2*x_data - 1
#        y_data = 2*y_data - 1 # with output mapping, easier to think of these as 0,1 ints

    # shuffle columns so not always same inputs matter
    perm = _get_perm(task) 
    x0 = np.where(perm == 0)[0][0] # index of X0 in permuted data
    x1 = np.where(perm == 1)[0][0]

    x_data = x_data[:, perm] 

#    #padding
#    x_data = np.concatenate([x_data, np.zeros([len(x_data), num_hidden_hyper - num_input])], axis=1)
#    y_data = np.concatenate([np.zeros([len(y_data), num_hidden_hyper - num_output]), y_data], axis=1)
    if cue_dimensions:
        cue_data = np.zeros_like(x_data)
        cue_data[:, [x0, x1]] = 1.
        x_data = np.concatenate([x_data, cue_data], axis=-1)

    dataset = {"x": x_data, "y": y_data, "relevant": [x0, x1]}

    return dataset 

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
        self.is_base_input = tf.placeholder_with_default(True, []) # whether is base input 
        self.is_base_output = tf.placeholder_with_default(True, []) # whether is base output:
                                                                    # i.e. to calculate loss as XE on output modes, or L2 over all 

        # base task input
        input_size = 2*num_input if cue_dimensions else num_input 
        self.base_input_ph = tf.placeholder(tf.float32, shape=[None, input_size])
        self.base_target_ph = tf.placeholder(tf.int32, shape=[None,])
        self.lr_ph = tf.placeholder(tf.float32)

        input_processing_1 = slim.fully_connected(self.base_input_ph, num_hidden, 
                                                  activation_fn=internal_nonlinearity) 

        processed_input = slim.fully_connected(input_processing_1, num_hidden_hyper, 
                                               activation_fn=internal_nonlinearity) 

        self.target_processor_nontf = random_orthogonal(num_hidden_hyper)[:, :2]
        self.target_processor = tf.constant(self.target_processor_nontf, dtype=tf.float32)

        target_one_hot = tf.one_hot(self.base_target_ph, 2)
        processed_targets = tf.matmul(target_one_hot, tf.transpose(self.target_processor)) 

        def output_mapping(X):
            """hidden space mapped back to T/F output logits"""
            res = tf.matmul(X, self.target_processor)
            return res

        # dummy fills for placeholders when they're switched off,
        # because tensorflow is silly
        self.dummy_base_input = np.zeros([batch_size, input_size])
        self.dummy_base_output = np.zeros([batch_size])
        self.dummy_meta_input = np.zeros([batch_size, num_hidden_hyper])
        self.dummy_meta_output = np.zeros([batch_size, num_hidden_hyper])

        # meta task input
        self.meta_input_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_target_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])

        processed_input = tf.cond(self.is_base_input,
            lambda: processed_input,
            lambda: self.meta_input_ph)
        processed_targets = tf.cond(self.is_base_output,
            lambda: processed_targets,
            lambda: self.meta_target_ph)
        
        # function embedding "guessing" network 
        self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess

        guess_input = tf.concat([processed_input, processed_targets], axis=-1)
        guess_input = tf.boolean_mask(guess_input, self.guess_input_mask_ph)

        guess_hidden_1 = slim.fully_connected(guess_input, num_hidden_hyper,
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
        task_hidden = processed_input 
        for i in range(num_task_hidden_layers):
            task_hidden = internal_nonlinearity(tf.matmul(task_hidden, self.hidden_weights[i]) + self.hidden_biases[i])
        self.raw_output = tf.matmul(task_hidden, Wfinal) + bfinal
        mapped_output = output_mapping(self.raw_output)
        self.base_output = tf.nn.softmax(mapped_output)

        self.base_loss = tf.cond(self.is_base_output,
            lambda: tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, 
                                                            logits=mapped_output),
            lambda: tf.reduce_sum(tf.square(self.raw_output - processed_targets), axis=1))
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        #base_full_optimizer = tf.train.MomentumOptimizer(self.lr_ph, train_momentum)
        base_full_optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        self.base_full_train = base_full_optimizer.minimize(self.total_base_loss)


        # initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    
#    def _guess_dataset(self, dataset):
#        return np.concatenate([dataset["x"], dataset["y"]],
#                              axis=1)
#
#    
    def _guess_mask(self, dataset_length):
        mask = np.zeros(dataset_length, dtype=np.bool)
        indices = np.random.permutation(dataset_length)[:meta_batch_size]
        mask[indices] = True
        return mask

    def dataset_eval(self, dataset, zeros=False, base_input=True, base_output=True):
        this_feed_dict = {
            self.base_input_ph: dataset["x"] if base_input else self.dummy_base_input,
            self.guess_input_mask_ph: np.zeros(len(dataset["x"]), dtype=np.bool) if zeros else np.ones(len(dataset["x"]), dtype=np.bool),
            self.is_base_input: base_input,
            self.is_base_output: base_output,
            self.base_target_ph: dataset["y"] if base_output else self.dummy_base_output,
            self.meta_input_ph: self.dummy_meta_input if base_input else dataset["x"],
            self.meta_target_ph: self.dummy_meta_output if base_output else dataset["y"]
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        return loss


    def dataset_embedding_eval(self, dataset, embedding, zeros=False, base_input=True, base_output=True, meta_binary=False):
        this_feed_dict = {
            self.embedding_is_fed: True,
            self.feed_embedding_ph: np.zeros_like(embedding) if zeros else embedding,
            self.base_input_ph: dataset["x"] if base_input else self.dummy_base_input,
            self.guess_input_mask_ph: np.zeros(len(dataset["x"]), dtype=np.bool) if zeros else np.ones(len(dataset["x"]), dtype=np.bool),
            self.is_base_input: base_input,
            self.is_base_output: base_output,
            self.base_target_ph: dataset["y"] if base_output or meta_binary else self.dummy_base_output,
            self.meta_input_ph: self.dummy_meta_input if base_input else dataset["x"],
            self.meta_target_ph: self.dummy_meta_output if base_output or meta_binary else dataset["y"]
        }
        loss = self.sess.run(self.total_base_loss, feed_dict=this_feed_dict)
        return loss


    def dataset_train_step(self, dataset, lr, base_input=True, base_output=True, meta_binary=False):
        guess_mask = self._guess_mask(len(dataset["x"]))
        this_feed_dict = {
            self.lr_ph: lr,
            self.base_input_ph: dataset["x"] if base_input else self.dummy_base_input,
            self.guess_input_mask_ph: guess_mask, 
            self.is_base_input: base_input,
            self.is_base_output: base_output,
            self.base_target_ph: dataset["y"] if base_output or meta_binary else self.dummy_base_output,
            self.meta_input_ph: self.dummy_meta_input if base_input else dataset["x"],
            self.meta_target_ph: self.dummy_meta_output if base_output or meta_binary else dataset["y"]
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
        elif meta_task[:2] == "is":
            pos_class = meta_task[2:]
            for task in self.base_tasks: 
                x_data.append(self.get_task_embedding(self.base_datasets[task])[0, :])
                task_type = task.split(";")[0]
                if task_type == pos_class:
                    y_data.append(1)
                else:
                    y_data.append(0)

        return {"x": np.array(x_data), "y": np.array(y_data)}


    def meta_true_eval(self):
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
                                                            base_input=False,
                                                            base_output=False)

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
                                                        base_input=False,
                                                        base_output=False)

                    names.append("ID:" + task + "->" + task)
                    losses.append(self.dataset_embedding_eval(dataset, mapped_embedding))

            else:
                print("Skipping meta true eval: " + meta_task + " (not implemented)") 
                continue

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
            losses[self.task_to_index[task] - offset] = self.dataset_eval(dataset,
                                                                          base_input=False, 
                                                                          base_output=len(dataset["y"].shape) == 1)

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

        for task in self.new_tasks:
            dataset =  self.new_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        for task in self.base_meta_tasks:
            dataset = self.meta_dataset_cache[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset,
                                                                 base_input=False,
                                                                 base_output=len(dataset["y"].shape) == 1)

        return losses

    def get_outputs(self, dataset, new_dataset=None, base_input=True, base_output=True, zeros=False):
        if new_dataset is not None:
            this_x = np.concatenate([dataset["x"], new_dataset["x"]], axis=0)
            dummy_y = np.zeros(len(new_dataset["x"])) if base_output else np.zeros_like(new_dataset["x"])
            this_y = np.concatenate([dataset["y"], dummy_y], axis=0)
            this_mask = np.zeros(len(this_x), dtype=np.bool)
            this_mask[:len(dataset["x"])] = 1. # use only these to guess
        else:
            this_x = dataset["x"]
            this_y = dataset["y"]
            this_mask = np.ones(len(dataset["x"]), dtype=np.bool)

        if zeros:
            this_mask = np.zeros_like(this_mask)

        this_feed_dict = {
            self.base_input_ph: this_x if base_input else self.dummy_base_input,
            self.guess_input_mask_ph: this_mask,
            self.is_base_input: base_input,
            self.is_base_output: base_output,
            self.base_target_ph: this_y if base_output else self.dummy_base_output,
            self.meta_input_ph: self.dummy_meta_input if base_input else this_x,
            self.meta_target_ph: self.dummy_meta_output if base_output else this_y
        }
        this_fetch = self.base_output if base_output else self.raw_output 
        outputs = self.sess.run(this_fetch, feed_dict=this_feed_dict)
        if base_output:
            outputs = 2*np.argmax(outputs, axis=-1) - 1
        if new_dataset is not None:
            outputs = outputs[len(dataset["x"]):, :]
        return outputs


    def new_outputs(self, new_task, zeros=False):
        """Returns outputs on a new task.
           zeros: if True, will give empty dataset to guessing net, for
           baseline"""
        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
        dataset = self.new_datasets[new_task_full_name]
        return self.get_outputs(dataset, zeros=zeros)


    def train_base_tasks(self, filename):
        """Train model to perform base tasks as meta task."""
        with open(filename, "w") as fout:
            fout.write("epoch, " + ", ".join(self.base_tasks + self.base_meta_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks + self.base_meta_tasks]) + "\n"

            learning_rate = init_learning_rate
            meta_learning_rate = init_meta_learning_rate

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
                    self.dataset_train_step(dataset, meta_learning_rate, 
                                            base_output=len(dataset["y"].shape) == 1,
                                            base_input=False)

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

                if epoch % lr_decays_every == 0 and epoch > 0 and meta_learning_rate > min_learning_rate:
                    meta_learning_rate *= meta_lr_decay


    def train_new_tasks(self, filename_prefix):
        print("Now training new tasks...")

        with open(filename_prefix + "new_losses.csv", "w") as fout:
            with open(filename_prefix + "meta_true_losses.csv", "w") as fout_meta:
                fout.write("epoch, " + ", ".join(self.all_tasks) + "\n")
                format_string = ", ".join(["%f" for _ in self.all_tasks]) + "\n"

                for new_task in self.new_tasks:
                    dataset = self.new_datasets[new_task]
                    with open(filename_prefix + new_task + "_outputs.csv", "w") as foutputs:
                        foutputs.write("type, " + ', '.join(["input%i" % i for i in range(len(dataset["y"]))]) + "\n")
                        foutputs.write("target, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(dataset["y"].flatten()) + "\n")

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
                meta_learning_rate = new_init_meta_learning_rate
                for epoch in range(1, max_new_epochs):
                    if epoch % refresh_meta_cache_every == 0:
                        for task in self.base_meta_tasks:
                            self.meta_dataset_cache[task] = self.get_meta_dataset(task)

                    order = np.random.permutation(self.num_tasks)
                    for task_i in order:
                        task = self.all_tasks[task_i]
                        base_input=True
                        base_output=True
                        this_lr = learning_rate
                        if task in self.new_tasks:
                            dataset =  self.new_datasets[task]
                        elif task in self.base_meta_tasks:
                            dataset = self.meta_dataset_cache[task]
                            base_input=False
                            base_output=len(dataset["y"].shape) == 1
                            this_lr = meta_learning_rate
                        else:
                            dataset =  self.base_datasets[task]
                        self.dataset_train_step(dataset, learning_rate, 
                                                base_input=base_input,
                                                base_output=base_output)

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

                    if epoch % lr_decays_every == 0 and epoch > 0 and meta_learning_rate > min_learning_rate:
                        meta_learning_rate *= meta_lr_decay

        for new_task in self.new_tasks:
            
            with open(filename_prefix + new_task + "_outputs.csv", "a") as foutputs:
                curr_net_outputs = self.new_outputs(new_task)
                foutputs.write("trained_emb, " + ', '.join(["%f" for i in range(len(curr_net_outputs))]) % tuple(curr_net_outputs) + "\n")


    def get_task_embedding(self, dataset, base_input=True, base_output=True):
        """Gets task embedding"""
        return self.sess.run(
            self.function_embedding,
            feed_dict={
                self.base_input_ph: dataset["x"] if base_input else self.dummy_base_input,
                self.guess_input_mask_ph: np.ones(len(dataset["x"]), dtype=np.bool),
                self.is_base_input: base_input,
                self.is_base_output: base_output,
                self.base_target_ph: dataset["y"] if base_output else self.dummy_base_output,
                self.meta_input_ph: self.dummy_meta_input if base_input else dataset["x"],
                self.meta_target_ph: self.dummy_meta_output if base_output else dataset["y"]
            }) 


    def save_embeddings(self, filename, meta_task=None):
        """Saves all task embeddings, if meta_task is not None first computes
           meta_task mapping on them."""
        def _simplify(t):
            split_t = t.split(';')
            return ';'.join([split_t[0], split_t[2]])
        with open(filename, "w") as fout:
            basic_tasks = self.base_tasks + self.new_tasks
            simplified_tasks = [_simplify(t) for t in basic_tasks]
            fout.write("dimension, " + ", ".join(simplified_tasks + self.base_meta_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.all_tasks]) + "\n"
            task_embeddings = np.zeros([len(self.all_tasks), num_hidden_hyper])

            for task in self.all_tasks:
                base_input = True
                base_output = True
                if task in self.new_tasks:
                    dataset =  self.new_datasets[task]
                elif task in self.base_tasks:
                    dataset =  self.base_datasets[task]
                else:
                    dataset = self.get_meta_dataset(task)
                    base_input = False 
                    base_output = len(dataset["y"].shape) == 1
                task_i = self.task_to_index[task] 
                task_embeddings[task_i, :] = self.get_task_embedding(dataset, 
                                                                     base_input=base_input,
                                                                     base_output=base_output)

            if meta_task is not None:
		meta_dataset = self.get_meta_dataset(meta_task)
		task_embeddings = self.get_outputs(meta_dataset,
						   {"x": task_embeddings},
						   base_input=False,
                                                   base_output=False)

            for i in range(num_hidden_hyper):
                fout.write(("%i, " %i) + (format_string % tuple(task_embeddings[:, i])))
                
## running stuff

for run_i in xrange(num_runs):
    np.random.seed(run_i)
    perm_list_dict = {task: (np.random.permutation(_get_perm_list_template(num_input)) if task not in ["XO", "NOTX0"] else np.random.permutation(_get_single_perm_list_template(num_input))) for task in total_tasks} 
    tf.set_random_seed(run_i)
    filename_prefix = "run%i" %(run_i)
    print("Now running %s" % filename_prefix)

    model = meta_model(num_input, base_tasks, base_task_repeats, new_tasks) 
    model.save_embeddings(filename=output_dir + filename_prefix + "_init_embeddings.csv")
    model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
    model.save_embeddings(filename=output_dir + filename_prefix + "_guess_embeddings.csv")
    for meta_task in base_meta_tasks:
        if meta_task[:2] == "is": # not a true meta mapping
            continue 
        model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_guess_embeddings.csv",
                              meta_task=meta_task)

    model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
    model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")
    for meta_task in base_meta_tasks:
        if meta_task[:2] == "is": # not a true meta mapping
            continue 
        model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_final_embeddings.csv",
                              meta_task=meta_task)


    tf.reset_default_graph()

