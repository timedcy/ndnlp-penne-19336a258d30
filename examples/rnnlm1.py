"""
Deep recurrent language model. This one stacks RNNs using the
recurrent.Stacked class, which is analogous to composition of
finite-state transducers.
"""

import sys
sys.path.append("..")
from penne import *
from penne import lm
from penne import recurrent
import numpy

hidden_dims = 100
depth = 1

data = lm.read_data("../data/inferno.txt")
#data = lm.read_data("../data/ptb.train.txt")
vocab = lm.make_vocab(data)
numberizer = lm.Numberizer(vocab)

layers = [recurrent.LSTM(hidden_dims, -len(vocab), hidden_dims)]
for i in xrange(depth-1):
    layers.append(recurrent.LSTM(hidden_dims, hidden_dims, hidden_dims))
rnn = recurrent.Stacked(*layers)

output_layer = make_layer(hidden_dims, len(vocab), f=logsoftmax)

trainer = SGD(learning_rate=0.01)

for epoch in xrange(1):
    epoch_loss = 0.
    n = 0
    for words in data:
        loss = constant(0.)
        prev_w = numberizer.numberize("<s>")
        rnn.start()
        for word in words:
            w = numberizer.numberize(word)
            o = output_layer(rnn.step(prev_w))
            loss -= o[w]
            prev_w = w
            
        sent_loss = trainer.receive(loss)
        epoch_loss += sent_loss
        n += len(words)
    print "epoch=%s ppl=%s" % (epoch, numpy.exp(epoch_loss/n))

