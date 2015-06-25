"""Recurrent neural networks as finite-state transducers."""

from expr import *
from nn import *
import numpy

class Transducer(object):
    """Base class for transducers."""
    def transduce(self, inps):
        self.start()
        outputs = []
        for inp in inps:
            outputs.append(self.step(inp))
        return outputs

class Stacked(Transducer):
    """Several stacked recurrent networks, or, the composition of several FSTs."""

    def __init__(self, *layers):
        self.layers = layers

    def start(self):
        for layer in self.layers:
            layer.start()
    def start_from(self, other):
        for layer, other_layer in zip(self.layers, other.layers):
            layer.start_from(other_layer)

    def step(self, inp):
        val = inp
        for layer in self.layers:
            val = layer.step(val)
        return val

class Simple(Transducer):
    """Simple (Elman) recurrent network.

    hidden_dims: Number of hidden units.
    input_dims:  Number of input units.
    output_dims: Number of output units. Must be equal to hidden_dims!
    f:           Activation function (default tanh)
    """

    def __init__(self, hidden_dims, input_dims, output_dims, f=tanh, model=parameter.all):
        if output_dims != hidden_dims:
            raise ValueError()
        self.layer = make_layer([hidden_dims, input_dims], hidden_dims, f=f, model=model)
        self.initial = constant(numpy.zeros((hidden_dims,))) # or parameter?
        
    def start(self):
        self.state = self.initial
    def start_from(self, other):
        self.state = other.state

    def step(self, inp):
        """inp can be either a vector Expression or an int """
        self.state = self.layer(self.state, inp)
        return self.state

class LSTM(Transducer):
    """Long short-term memory recurrent network.

    This version is from: Alex Graves, "Generating sequences with
    recurrent neural networks." arXiv:1308.0850.

    hidden_dims: Number of hidden units.
    input_dims:  Number of input units.
    output_dims: Number of output units.
    f:           Activation function (default tanh)
    """

    def __init__(self, hidden_dims, input_dims, output_dims, model=parameter.all):
        dims = [input_dims, hidden_dims, hidden_dims]
        self.input_gate = make_layer(dims, hidden_dims, f=sigmoid, model=model)
        self.forget_gate = make_layer(dims, hidden_dims, f=sigmoid, model=model)
        self.output_gate = make_layer(dims, hidden_dims, f=sigmoid, model=model)
        self.input_layer = make_layer(dims[:-1], hidden_dims, f=tanh, model=model)
        self.h0 = constant(numpy.zeros((hidden_dims,))) # or parameter?
        self.c0 = constant(numpy.zeros((hidden_dims,))) # or parameter?

    def start(self):
        self.h = self.h0
        self.c = self.c0
    def start_from(self, other):
        self.h = other.h
        self.c = other.c

    def step(self, inp):
        i = self.input_gate(inp, self.h, self.c)
        f = self.forget_gate(inp, self.h, self.c)
        self.c = f * self.c + i * self.input_layer(inp, self.h)
        o = self.output_gate(inp, self.h, self.c)
        self.h = o * tanh(self.c)
        return o
