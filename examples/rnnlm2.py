"""
Another implementation of a deep recurrent language model. This one
stacks RNNs by computing the entire output sequence of one RNN before
feeding to the next RNN up.
"""

import sys, time
sys.path.append("..")
from penne import *
from penne import lm
from penne import recurrent
import numpy

hidden_dims = 100
depth = 1

#data = lm.read_data("../data/inferno.txt")
data = lm.read_data("../data/ptb.train.txt")[:420]
vocab = lm.make_vocab(data)
numberizer = lm.Numberizer(vocab)

layers = [recurrent.LSTM(hidden_dims, -len(vocab), hidden_dims)]
for i in xrange(depth-1):
    layers.append(recurrent.LSTM(hidden_dims, hidden_dims, hidden_dims))
output_layer = make_layer(hidden_dims, len(vocab), f=logsoftmax)

trainer = SGD(learning_rate=0.01)

for epoch in xrange(1):
    epoch_loss = 0.
    n = 0
    for iteration, words in enumerate(data):
        nums = [numberizer.numberize(word) for word in ["<s>"]+words]

        xs = nums[:-1]
        for layer in layers:
            xs = layer.transduce(xs)

        # Compute all the output layers at once
        o = output_layer(stack(xs))
        w = stack([one_hot(len(vocab), num) for num in nums[1:]])
        loss = -einsum("ij,ij->", w, o)

        sent_loss = trainer.receive(loss)
        epoch_loss += sent_loss
        n += len(words)
    print "epoch=%s ppl=%s" % (epoch, numpy.exp(epoch_loss/n))

