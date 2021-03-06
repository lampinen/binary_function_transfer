from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

#import cProfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from scipy.spatial.distance import pdist
import datasets
from orthogonal_matrices import random_orthogonal

pi = np.pi
### Parameters
num_input = 10
num_output = 1 # cannot be changed without somewhat substantial code modifications
num_hidden = 64
num_hidden_hyper = 64
num_runs = 10 
run_offset = 0
init_learning_rate = 1e-4
init_meta_learning_rate = 2e-4
new_init_learning_rate = 1e-6
new_init_meta_learning_rate = 1e-6
lr_decay = 0.85
meta_lr_decay = 0.85
lr_decays_every = 100
min_learning_rate = 1e-6
refresh_meta_cache_every = 1 # how many epochs between updates to meta_dataset_cache

#train_momentum = 0.8
#adam_epsilon = 1e-3

max_base_epochs = 4000 
max_new_epochs = 200
num_task_hidden_layers = 3
num_meta_hidden_layers = 3
output_dir = "meta_results/"
save_every = 10 #20
tf_pm = True # if true, code t/f as +/- 1 rather than 1/0
cue_dimensions = True # if true, provide two-hot cues of which dimensions are relevant
base_hard_eval = True # eval on accuracy on thresholded values, which is bounded and more interpretable than XE loss.
save_input_embeddings = False # whether to save full input embeddings -- is pretty space expensive
hyper_convolutional = False # whether hyper network creates weights convolutionally
conv_in_channels = 6

batch_size = 1024
meta_batch_size = 768 # how much of each dataset the function embedding guesser sees 
early_stopping_thresh = 0.005
base_tasks = ["X0", "NOTX0", "AND", "NOTAND", "OR", "XOR", "NOTXOR"]
base_meta_tasks = ["isX0", "isNOTX0", "isAND", "isNOTAND", "isOR", "isXOR", "isNOTXOR"]
new_meta_tasks = ["isNOTOR"]
base_meta_mappings = ["ID", "NOT", "A2O"]#, "NOTA2O"]
base_task_repeats = 44 # how many times each base task is seen
new_tasks = ["X0", "AND", "OR", "NOTOR", "NOTAND", "XOR", "NOTXOR"]
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


def _get_meta_mappings(this_base_tasks, this_base_meta_tasks, this_base_meta_mappings, this_new_tasks=None, include_new=False):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_base_meta_tasks = this_base_meta_tasks + this_base_meta_mappings
    meta_mappings = {meta_task: {"base": [], "meta": []} for meta_task in all_base_meta_tasks}
    if include_new:
        basic_tasks = this_base_tasks + this_new_tasks
    else:
        basic_tasks = this_base_tasks
    for meta_task in all_base_meta_tasks:
        if meta_task == "NOT":
            for task in basic_tasks: 
                stripped_task = ";".join(task.split(";")[:-1])
                other = "NOT" + stripped_task if task[:3] != "NOT" else stripped_task[3:]
                other_tasks = [t for t in basic_tasks if ";".join(t.split(";")[:-1]) == other]
                if other_tasks != []:
                    other = other_tasks[0]
                    meta_mappings[meta_task]["base"].append((task, other))
            for task in this_base_meta_tasks:
                other = task[:2] + task[5:] if "NOT" in task else task[:2] + "NOT" + task[2:]
                if other in this_base_meta_tasks:
                    meta_mappings[meta_task]["meta"].append((task, other))
        elif meta_task == "A2O":
            for task in basic_tasks: 
                stripped_task = ";".join(task.split(";")[:-1])
                if "XOR" in stripped_task:
                    other = "NOT" + stripped_task if task[:3] != "NOT" else stripped_task[3:]
                elif "OR" in stripped_task:
                    other = "NOTAND" + stripped_task[5:] if task[:3] == "NOT" else "AND" + stripped_task[2:]
                elif "AND" in stripped_task:
                    other = "NOTOR" + stripped_task[6:] if task[:3] == "NOT" else "OR" + stripped_task[3:]
                else: # identity on all other tasks
                    meta_mappings[meta_task]["base"].append((task, task))
                    continue
                    
                other_tasks = [t for t in basic_tasks if ";".join(t.split(";")[:-1]) == other]
                if other_tasks != []:
                    other = other_tasks[0]
                    meta_mappings[meta_task]["base"].append((task, other))

            for task in this_base_meta_tasks:
                if "XOR" in task:
                    other = task[:2] + task[5:] if "NOT" in task else task[:2] + "NOT" + task[2:]
                elif "OR" in task:
                    other = "isNOTAND" if task[2:5] == "NOT" else "isAND"
                elif "AND" in task:
                    other = "isNOTOR" if task[2:5] == "NOT" else "isOR"
                else: # identity on all other tasks
                    meta_mappings[meta_task]["meta"].append((task, task))
                    continue

                if other in this_base_meta_tasks:
                    meta_mappings[meta_task]["meta"].append((task, other))
        elif meta_task == "NOTA2O":
            for task in basic_tasks: 
                stripped_task = ";".join(task.split(";")[:-1])
                if "XOR" in stripped_task:
                    meta_mappings[meta_task]["base"].append((task, task))
                    continue
                elif "OR" in stripped_task:
                    other = "AND" + stripped_task[5:] if task[:3] == "NOT" else "NOTAND" + stripped_task[2:]
                elif "AND" in stripped_task:
                    other = "OR" + stripped_task[6:] if task[:3] == "NOT" else "NOTOR" + stripped_task[3:]
                elif "X0" in stripped_task:
                    other = stripped_task[3:] if task[:3] == "NOT" else "NOT" + stripped_task
                else: 
                    raise ValueError("Unknown base task for NOTA2O" + task)
                    
                other_tasks = [t for t in basic_tasks if ";".join(t.split(";")[:-1]) == other]
                if other_tasks != []:
                    other = other_tasks[0]
                    meta_mappings[meta_task]["base"].append((task, other))

            for task in this_base_meta_tasks:
                if "XOR" in task:
                    meta_mappings[meta_task]["meta"].append((task, task))
                    continue
                elif "OR" in task:
                    other = "isAND" if task[2:5] == "NOT" else "isNOTAND"
                elif "AND" in task:
                    other = "isOR" if task[2:5] == "NOT" else "isNOTOR"
                elif "X0" in task:
                    other = "isX0" if task[2:5] == "NOT" else "isNOTX0"
                else: 
                    raise ValueError("Unknown meta task for NOTA2O: " + task)

                if other in this_base_meta_tasks:
                    meta_mappings[meta_task]["meta"].append((task, other))
        elif meta_task == "ID":
            for task in basic_tasks: 
                meta_mappings[meta_task]["base"].append((task, task))
            for task in this_base_meta_tasks:
                meta_mappings[meta_task]["meta"].append((task, task))
        elif meta_task[:2] == "is":
            pos_class = meta_task[2:]
            for task in basic_tasks: 
                task_type = task.split(";")[0]
                meta_mappings[meta_task]["base"].append((task, 1*(task_type == pos_class)))

    return meta_mappings


class meta_model(object):
    """A meta-learning model for binary functions."""
    def __init__(self, num_input, base_tasks, base_task_repeats, new_tasks,
                 base_meta_tasks, base_meta_mappings, new_meta_tasks,
                 meta_two_level=True):
        """args:
            num_input: number of binary inputs
            base_tasks: list of base binary functions to compute
            base_meta_tasks: tasks whose domain is functions but range is not
            base_meta_mappings: tasks whose domain AND range are both function
                embeddings
            base_task_repeats: how many times to repeat each task (must be <= 
                num_input choose 2)
            new_tasks: new tasks to test on
            new_meta_tasks: new meta tasks
            meta_two_level: If true, the meta tasks will be trained to make
                mappings like NOT: isAND -> isNOTAND, which is a more abstract
                sense of NOT than is implied by AND -> NOTAND (the most obvious
                extension of this would be NOT: isAND -> NOTisAND). If false, 
                meta tasks are only trained on base tasks.
        """
        self.num_input = num_input
        self.num_output = num_output
        self.meta_two_level = meta_two_level

        # base datasets
        self.base_tasks = []
        self.base_datasets = {}

        for task_i, task_name in enumerate(base_tasks * base_task_repeats):
            dataset = _get_dataset(task_name, num_input)
            task_full_name = task_name + ";" + str(dataset["relevant"][0]) + str(dataset["relevant"][1]) + ";old" + str(task_i)
            self.base_datasets[task_full_name] = dataset 
            self.base_tasks.append(task_full_name) 

        self.base_meta_tasks = base_meta_tasks 
        self.base_meta_mappings = base_meta_mappings
        self.all_base_meta_tasks = base_meta_tasks + base_meta_mappings
        self.new_meta_tasks = new_meta_tasks 
        self.meta_dataset_cache = {t: {} for t in self.all_base_meta_tasks + self.new_meta_tasks} # will cache datasets for a certain number of epochs
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

        self.all_tasks = self.base_tasks + self.new_tasks + self.all_base_meta_tasks
        self.num_tasks = num_tasks = len(self.all_tasks)
        self.task_to_index = dict(zip(self.all_tasks, range(num_tasks)))

        self.meta_mappings_base = _get_meta_mappings(
            self.base_tasks, self.base_meta_tasks, self.base_meta_mappings)

        self.meta_mappings_full = _get_meta_mappings(
            self.base_tasks, self.base_meta_tasks + self.new_meta_tasks,
            self.base_meta_mappings, self.new_tasks,
            include_new=True)

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
        self.processed_input = processed_input
        processed_targets = tf.cond(self.is_base_output,
            lambda: processed_targets,
            lambda: self.meta_target_ph)
        self.processed_targets = processed_targets
        
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

        self.loss = tf.cond(self.is_base_output,
            lambda: tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, 
                                                            logits=mapped_output),
            lambda: tf.reduce_sum(tf.square(self.raw_output - processed_targets), axis=1))
        self.base_hard_loss = tf.cast(
            tf.logical_not(tf.equal(tf.argmax(target_one_hot, axis=-1),
                                    tf.argmax(mapped_output, axis=-1))),
                           tf.float32)

        self.total_loss = tf.reduce_mean(self.loss)
        self.total_base_hard_loss = tf.reduce_mean(self.base_hard_loss)
        #base_full_optimizer = tf.train.MomentumOptimizer(self.lr_ph, train_momentum)
        base_full_optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        self.base_full_train = base_full_optimizer.minimize(self.total_loss)


        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.refresh_meta_dataset_cache()
        
    
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
        fetch  = self.total_base_hard_loss if base_output and base_hard_eval else self.total_loss 
        loss = self.sess.run(fetch, feed_dict=this_feed_dict)
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
        fetch  = self.total_base_hard_loss if base_output and base_hard_eval else self.total_loss 
        loss = self.sess.run(fetch, feed_dict=this_feed_dict)
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
        

    def get_meta_dataset(self, meta_task, include_new=False, override_m2l=False):
        """override_m2l is used to allow meta mapped class. even if not trained on it"""
        x_data = []
        y_data = []
        if include_new:
            this_base_tasks = self.meta_mappings_full[meta_task]["base"]
            this_meta_tasks = self.meta_mappings_full[meta_task]["meta"]
        else:
            this_base_tasks = self.meta_mappings_base[meta_task]["base"]
            this_meta_tasks = self.meta_mappings_base[meta_task]["meta"]
        for (task, other) in this_base_tasks:
            task_dataset = self.base_datasets[task] if task in self.base_tasks else self.new_datasets[task]
            x_data.append(self.get_task_embedding(task_dataset)[0, :])
            if other in [0, 1]:  # for classification meta tasks
                y_data.append(other)
            else:
                other_dataset = self.base_datasets[other] if other in self.base_tasks else self.new_datasets[other]
                y_data.append(self.get_task_embedding(other_dataset)[0, :])
        if self.meta_two_level or override_m2l:
            for (task, other) in this_meta_tasks:
                embedding = self.get_task_embedding(self.meta_dataset_cache[task], 
                                                    base_input=False)[0, :]
                other_embedding = self.get_task_embedding(self.meta_dataset_cache[other], 
                                                          base_input=False)[0, :]
                x_data.append(embedding)
                y_data.append(other_embedding)
        return {"x": np.array(x_data), "y": np.array(y_data)}


    def meta_true_eval(self):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        losses = []
        names = []
        for meta_task in self.base_meta_mappings:
            meta_dataset = self.get_meta_dataset(meta_task)
            for task, other in self.meta_mappings_full[meta_task]["base"]:
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

                names.append(meta_task + ":" + task + "->" + other)
                losses.append(self.dataset_embedding_eval(dataset, mapped_embedding))

        return losses, names


    def meta_mapped_classification_true_eval(self):
        """Evaluates true meta loss of classification embeddings output 
           by the embedding output by the meta mapping tasks"""
        losses = []
        names = []
        for meta_task in self.base_meta_mappings:
            meta_dataset = self.get_meta_dataset(meta_task, include_new=True,
                                                 override_m2l=True)
            for task, other in self.meta_mappings_full[meta_task]["meta"]:
                task_dataset = self.get_meta_dataset(task, include_new=True)

                task_embedding = self.get_task_embedding(task_dataset,
                                                         base_input=False)

                dataset = self.get_meta_dataset(other, include_new=True)

                mapped_embedding = self.get_outputs(meta_dataset,
                                                    {"x": task_embedding},
                                                    base_input=False,
                                                    base_output=False)

                names.append(meta_task + ":" + task + "->" + other)
                losses.append(
                    self.dataset_embedding_eval(dataset, mapped_embedding,
                                                base_input=False))

        return losses, names


    def base_eval(self):
        """Evaluates loss on the base tasks."""
        losses = np.zeros([len(self.base_tasks) + len(self.all_base_meta_tasks)])
        for task in self.base_tasks:
            dataset =  self.base_datasets[task]
            losses[self.task_to_index[task]] = self.dataset_eval(dataset)

        offset = len(self.new_tasks) # new come before meta in indices
        for task in self.all_base_meta_tasks:
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

        for task in self.all_base_meta_tasks:
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
            fout.write("epoch, " + ", ".join(self.base_tasks + self.all_base_meta_tasks) + "\n")
            format_string = ", ".join(["%f" for _ in self.base_tasks + self.all_base_meta_tasks]) + "\n"

            learning_rate = init_learning_rate
            meta_learning_rate = init_meta_learning_rate

            for epoch in range(max_base_epochs):

                if epoch % refresh_meta_cache_every == 0:
                    self.refresh_meta_dataset_cache()

                order = np.random.permutation(len(self.base_tasks))
                for task_i in order:
                    task = self.base_tasks[task_i]
                    dataset =  self.base_datasets[task]
                    self.dataset_train_step(dataset, learning_rate)

                order = np.random.permutation(len(self.all_base_meta_tasks))
                for task_i in order:
                    task = self.all_base_meta_tasks[task_i]
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


    def refresh_meta_dataset_cache(self, refresh_new=False):
        for task in self.base_meta_tasks:
            self.meta_dataset_cache[task] = self.get_meta_dataset(task)
        for task in self.base_meta_mappings:
            self.meta_dataset_cache[task] = self.get_meta_dataset(task)
        if refresh_new:
            for task in self.new_meta_tasks:
                self.meta_dataset_cache[task] = self.get_meta_dataset(task, include_new=True)


    def train_new_tasks(self, filename_prefix):
        print("Now training new tasks...")

        with open(filename_prefix + "new_losses.csv", "w") as fout:
            with open(filename_prefix + "meta_true_losses.csv", "w") as fout_meta:
                with open(filename_prefix + "meta_mapped_classification_true_losses.csv", "w") as fout_meta2:
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

                    self.refresh_meta_dataset_cache(refresh_new=True)

                    curr_meta_true_losses, meta_true_names = self.meta_true_eval() 
                    fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")
                    meta_format_string = ", ".join(["%f" for _ in meta_true_names]) + "\n"
                    curr_meta_output = ("0, ") + (meta_format_string % tuple(curr_meta_true_losses))
                    fout_meta.write(curr_meta_output)

                    curr_meta2_true_losses, meta2_true_names = self.meta_mapped_classification_true_eval() 
                    fout_meta2.write("epoch, " + ", ".join(meta2_true_names) + "\n")
                    meta2_format_string = ", ".join(["%f" for _ in meta2_true_names]) + "\n"
                    curr_meta2_output = ("0, ") + (meta2_format_string % tuple(curr_meta2_true_losses))
                    fout_meta2.write(curr_meta2_output)

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
                            self.refresh_meta_dataset_cache(refresh_new=True)

                        order = np.random.permutation(self.num_tasks)
                        for task_i in order:
                            task = self.all_tasks[task_i]
                            base_input=True
                            base_output=True
                            this_lr = learning_rate
                            if task in self.new_tasks:
                                dataset =  self.new_datasets[task]
                            elif task in self.all_base_meta_tasks:
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

                            curr_meta2_true_losses, _ = self.meta_mapped_classification_true_eval() 
                            curr_meta2_output = ("%i, " % epoch) + (meta2_format_string % tuple(curr_meta2_true_losses))
                            fout_meta2.write(curr_meta2_output)

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
            fout.write("dimension, " + ", ".join(simplified_tasks + self.all_base_meta_tasks) + "\n")
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


    def save_input_embeddings(self, filename):
        """Saves embeddings of all possible inputs."""
        raw_data, _ = datasets.X0_dataset(num_input)
        if tf_pm:
            raw_data = 2*raw_data - 1
        names = [("%i" * num_input) % tuple(raw_data[i, :]) for i in range(len(raw_data))]
        format_string = ", ".join(["%f" for _ in names]) + "\n"
        with open(filename, "w") as fout:
            fout.write("dimension, x0, x1, " + ", ".join(names) + "\n")
            for perm in _get_perm_list_template(num_input):
                this_perm = np.array(perm)
                x_data = raw_data.copy()
                x0 = np.where(this_perm == 0)[0] # index of X0 in permuted data
                x1 = np.where(this_perm == 1)[0]

                x_data = x_data[:, this_perm] 
                
                if cue_dimensions:
                    cue_data = np.zeros_like(x_data)
                    cue_data[:, [x0, x1]] = 1.
                    x_data = np.concatenate([x_data, cue_data], axis=-1)

                input_embeddings = self.sess.run(self.processed_input,
                                                 feed_dict = {
                                                     self.base_input_ph: x_data,
                                                     self.is_base_input: True,
                                                     self.base_target_ph: self.dummy_base_output,
                                                     self.meta_input_ph: self.dummy_meta_input,
                                                     self.meta_target_ph: self.dummy_meta_output
                                                     })
                
                for i in range(num_hidden_hyper):
                    fout.write(("%i, %i, %i, " %(i, x0, x1)) + (format_string % tuple(input_embeddings[:, i])))
                
## running stuff

for run_i in range(run_offset, run_offset+num_runs):
    for meta_two_level in [True, False]: 
        np.random.seed(run_i)
        perm_list_dict = {task: (np.random.permutation(_get_perm_list_template(num_input)) if task not in ["XO", "NOTX0"] else np.random.permutation(_get_single_perm_list_template(num_input))) for task in total_tasks} 
        tf.set_random_seed(run_i)
        filename_prefix = "m2l%r_run%i" %(meta_two_level, run_i)
        print("Now running %s" % filename_prefix)

        model = meta_model(num_input, base_tasks, base_task_repeats, new_tasks,
                           base_meta_tasks, base_meta_mappings, new_meta_tasks,
                           meta_two_level=meta_two_level) 
        model.save_embeddings(filename=output_dir + filename_prefix + "_init_embeddings.csv")

#        cProfile.run('model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")')
#        exit()
        model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
        model.save_embeddings(filename=output_dir + filename_prefix + "_guess_embeddings.csv")
        if save_input_embeddings:
            model.save_input_embeddings(filename=output_dir + filename_prefix + "_guess_input_embeddings.csv")
        for meta_task in base_meta_mappings:
            model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_guess_embeddings.csv",
                                  meta_task=meta_task)

        model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
        model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")
        for meta_task in base_meta_mappings:
            model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_final_embeddings.csv",
                                  meta_task=meta_task)


        tf.reset_default_graph()

