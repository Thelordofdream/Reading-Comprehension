# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow import constant
from tensorflow.contrib import rnn
import numpy as np


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
test_iters = 1

# Network Parameters
n_input_d = 28
n_input_q = 28
n_steps_d = 28  # timesteps
n_steps_q = 28
x_dim = 28
n_hidden = 16  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)

d = tf.placeholder("float", [None, n_steps_d, n_input_d])
q = tf.placeholder("float", [None, n_steps_q, n_input_q])
istate_fw_d = tf.placeholder("float", [None, 2 * n_hidden])
istate_bw_d = tf.placeholder("float", [None, 2 * n_hidden])
istate_fw_q = tf.placeholder("float", [None, 2 * n_hidden])
istate_bw_q = tf.placeholder("float", [None, 2 * n_hidden])
keep_prob_q = tf.placeholder(tf.float32)
keep_prob_d = tf.placeholder(tf.float32)
a = tf.placeholder("float", [None, n_classes])

# Define weights
weights_d = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([2 * n_input_d, 2 * n_hidden]), name='hidden_w_d'),
    'Wym': tf.Variable(tf.random_normal([2 * n_hidden, 2 * n_hidden]), name='Wym'),
    'Wum': tf.Variable(tf.random_normal([2 * n_hidden, 2 * n_hidden]), name='Wum'),
    'Wms': tf.Variable(tf.random_normal([2 * n_hidden, 1]), name='Wms'),
    'Wrg': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]), name='Wrg'),
    'Wug': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]), name='Wug'),
    'test': tf.Variable(tf.random_normal([2 * n_input_q, n_hidden]), name='test'),
}

biases_d = {
    'hidden': tf.Variable(tf.random_normal([2 * n_hidden]), name='hidden_b_d'),
}

weights_q = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input_q, n_hidden]), name='hidden_w_q'),
    'test': tf.Variable(tf.random_normal([2 * n_input_q, n_hidden]), name='test'),
}

biases_q = {
    'hidden': tf.Variable(tf.random_normal([n_hidden]), name='hidden_b_q'),
}


def create_network(name="N1", X=None, n_input=1, seq_len=1):
    with tf.variable_scope(name):
        # Linear activation
        # Forward direction cell
        with tf.device("/cpu:0"):
            lstm_fw_cell_q = rnn.LSTMCell(n_input, forget_bias=0.1, state_is_tuple=True)
            lstm_fw_cell_q = rnn.DropoutWrapper(lstm_fw_cell_q, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        # Backward direction cell
        with tf.device("/cpu:1"):
            lstm_bw_cell_q = rnn.LSTMCell(n_input, forget_bias=0., state_is_tuple=True)
            lstm_bw_cell_q = rnn.DropoutWrapper(lstm_bw_cell_q, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        # Get lstm cell output
        outputs, output1, output2 = rnn.static_bidirectional_rnn(lstm_fw_cell_q, lstm_bw_cell_q, X,
                                                                 initial_state_fw=lstm_fw_cell_q.zero_state(batch_size,
                                                                                                            tf.float32),
                                                                 initial_state_bw=lstm_bw_cell_q.zero_state(batch_size,
                                                                                                            tf.float32),
                                                                 sequence_length=seq_len)
        return outputs, output1, output2


def hidden_layer(name="N1", X=None, weights=None, biases=None, n_input=1, seq_len=1):
    with tf.variable_scope(name):
        # Reshape to prepare input to hidden activation
        X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
        Y = tf.matmul(X, weights) + biases
        Y = tf.split(Y, seq_len, 0)
        return Y


def shape_tranform(X=None, n_input=1, n_step=1):
    X = tf.transpose(X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    X = tf.split(X, n_step, 0)  # n_steps * (batch_size, n_hidden)
    return X


def Bi_LSTM_D(_X, _weights, _biases, _batch_size, _seq_len, _n_input):
    n_step = _seq_len
    _X = shape_tranform(X=_X, n_input=_n_input, n_step=n_step)
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.float32))

    outputs, output1, output2 = create_network(name="LSTM_D", X=_X, n_input=_n_input, seq_len=_seq_len)
    o1 = hidden_layer(name="dh", X=outputs, weights=_weights['hidden'], biases=_biases['hidden'], n_input=2 * _n_input,
                      seq_len=n_step)

    y = tf.nn.dropout(o1, keep_prob_q)
    return y


def Bi_LSTM_Q(_X, _weights, _biases, _batch_size, _seq_len, _n_input):
    n_step = _seq_len
    _X = shape_tranform(X=_X, n_input=_n_input, n_step=n_step)
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.float32))

    outputs, output1, output2 = create_network(name="LSTM_Q", X=_X, n_input=_n_input, seq_len=_seq_len)
    output1 = hidden_layer(name="qh1", X=output1, weights=_weights['hidden'], biases=_biases['hidden'],
                           n_input=_n_input,
                           seq_len=2)
    output2 = hidden_layer(name="qh2", X=output2, weights=_weights['hidden'], biases=_biases['hidden'],
                           n_input=_n_input,
                           seq_len=2)

    output = tf.concat([output1[0], output2[0]], 1)
    u = tf.nn.dropout(output, keep_prob_q)
    return u


y = Bi_LSTM_D(d, weights_d, biases_d, batch_size, n_steps_d, n_input_d)
u = Bi_LSTM_Q(q, weights_q, biases_q, batch_size, n_steps_q, n_input_q)
mu = tf.matmul(u, weights_d['Wum'])
m = []
for i in range(n_steps_d):
    m.append(tf.nn.tanh(tf.matmul(y[i], weights_d['Wym']) + mu))
m = tf.reshape(m, [-1, 2 * n_hidden])
s = tf.nn.softmax(tf.exp(tf.matmul(m, weights_d['Wms'])))
s = tf.split(s, n_steps_d, 0)
y = tf.transpose(y, [1, 0, 2])
s = tf.transpose(s, [1, 0, 2])
r = []
for i in range(batch_size):
    r.append(tf.transpose(tf.matmul(tf.transpose(y[i]), s[i])))
r = tf.reshape(r, [-1, 2 * n_hidden])
r_drop = tf.nn.dropout(r, keep_prob_d)
u_drop = tf.nn.dropout(u, keep_prob_q)
g = tf.nn.tanh(tf.matmul(u, weights_d['Wug']) + tf.matmul(r, weights_d['Wrg']))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=g))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

correct_pred = tf.equal(tf.argmax(g, 1), tf.argmax(a, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, "./Model for mnist/model.ckpt")
    for i in range(test_iters):
        # 持续迭代
        batch_d, batch_q = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_d = batch_d.reshape((batch_size, n_steps_d, n_input_d))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={d: batch_d, q: batch_d, a: batch_q,
                                                                 istate_fw_d: np.zeros((batch_size, 2 * n_hidden)),
                                                                 istate_bw_d: np.zeros((batch_size, 2 * n_hidden)),
                                                                 istate_fw_q: np.zeros((batch_size, 2 * n_hidden)),
                                                                 istate_bw_q: np.zeros((batch_size, 2 * n_hidden)),
                                                                 keep_prob_d: 1.0, keep_prob_q: 1.0}))
        print("Prediction:", sess.run(tf.argmax(g, 1), feed_dict={d: batch_d, q: batch_d,
                                                                  istate_fw_d: np.zeros((batch_size, 2 * n_hidden)),
                                                                  istate_bw_d: np.zeros((batch_size, 2 * n_hidden)),
                                                                  istate_fw_q: np.zeros((batch_size, 2 * n_hidden)),
                                                                  istate_bw_q: np.zeros((batch_size, 2 * n_hidden)),
                                                                  keep_prob_d: 1.0, keep_prob_q: 1.0}))
        print("Label:", sess.run(tf.argmax(batch_q, 1)))
        # print("G:", sess.run(g, feed_dict={d: batch_d, q: batch_d,
        #                                    istate_fw_d: np.zeros((batch_size, 2 * n_hidden)),
        #                                    istate_bw_d: np.zeros((batch_size, 2 * n_hidden)),
        #                                    istate_fw_q: np.zeros((batch_size, 2 * n_hidden)),
        #                                    istate_bw_q: np.zeros((batch_size, 2 * n_hidden)),
        #                                    keep_prob_d: 1.0, keep_prob_q: 1.0}))
