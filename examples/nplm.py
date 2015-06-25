"""
Feedforward language model.
Bengio et al., 2003. A neural probabilistic language model. JMLR 3:1137-1155.
"""

import sys
sys.path.append("..")
from penne import *
from penne import lm
import numpy

order = 4
embedding_dims = 100
hidden_dims = 100

data = lm.read_data("../data/inferno.txt")
#data = lm.read_data("../data/ptb.train.txt")
vocab = lm.make_vocab(data)
numberizer = lm.Numberizer(vocab)

input_layer = make_layer(-len(vocab), embedding_dims, f=None, bias=False)
hidden1_layer = make_layer(embedding_dims*(order-1), hidden_dims)
hidden2_layer = make_layer(hidden_dims, embedding_dims)
output_layer = make_layer(embedding_dims, len(vocab))

def prob(context):
    i = []
    for word in context:
        w = numberizer.numberize(word)
        e = input_layer(w)
        i.append(e)
    h1 = tanh(hidden1_layer(concatenate(i)))
    h2 = tanh(hidden2_layer(h1))
    return logsoftmax(output_layer(h2))

trainer = SGD(learning_rate=0.01)

for epoch in xrange(10):
    epoch_loss = 0.
    n = 0
    for ngram in lm.ngrams(data, order):
        p = prob(ngram[:-1])
        correct = one_hot(len(vocab), numberizer.numberize(ngram[-1]))
        loss = crossentropy(p, correct)
        epoch_loss += trainer.receive(loss)
        n += 1
    print "epoch=%s ppl=%s" % (epoch, numpy.exp(epoch_loss/n))
