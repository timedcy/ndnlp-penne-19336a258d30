"""Special expressions for neural networks."""

__all__ = ['sigmoid', 'rectify', 'hardtanh', 'logsoftmax', 'crossentropy', 'distance2', 'make_layer']

import numpy
from expr import *

## Activation functions

class sigmoid(Unary):
    """Logistic sigmoid function."""
    @staticmethod 
    def f(x):      
        with numpy.errstate(over='ignore'): 
            return 1./(1.+numpy.exp(-x))
    @staticmethod 
    def dfdx(x, y): return y*(1.-y)

class rectify(Unary):
    """Rectified linear unit = max(0, x)."""
    @staticmethod 
    def f(x):       return numpy.maximum(x, 0.)
    @staticmethod 
    def dfdx(x, y): return numpy.where(x > 0., 1., 0.)

class hardtanh(Unary):
    """Hard tanh function = clip(x, -1, 1)."""
    @staticmethod 
    def f(x):       return numpy.clip(x, -1., 1.)
    @staticmethod 
    def dfdx(x, y): return numpy.where(numpy.logical_and(-1. < x, x < 1.), 1., 0.)

class logsoftmax(Expression):
    """Log of the softmax function.

    softmax_i(x) = \exp x_i / \sum_j \exp x_j.
    The log is for better numerical stability when used with crossentropy.
    
    axis: along which to perform the softmax (default is last).
    """
    def __init__(self, arg, axis=-1):
        Expression.__init__(self, arg)
        self.axis = axis

    def forward(self, values):
        axis = self.axis
        v = values[self.args[0]]
        v = v - numpy.max(v, axis=axis, keepdims=True)
        values[self] = v - numpy.log(numpy.sum(numpy.exp(v), axis=axis, keepdims=True))

    def backward(self, values, gradients):
        axis = self.axis
        arg = self.args[0]
        if arg in gradients:
            gradients[arg] += gradients[self] 
            gradients[arg] -= numpy.sum(gradients[self], axis=axis, keepdims=True) * numpy.exp(values[self])

### Loss functions

def crossentropy(logp, correct):
    """Cross-entropy.

    logp: (expression evaluating to) a vector of log-probabilities
    correct: (expression evaluating to) the observed probabilities
    """
    return -dot(logp, correct)

def distance2(x, y):
    """Squared Euclidean distance."""
    d = x - y
    return dot(d, d)

### Fully-connected layer

def guess_gain(f, d):
    """Try to figure out how the activation function affects the
    variance of inputs/gradients."""

    if f is None: return 1.

    # Special case for the one non-componentwise activation function.
    # We could do this automatically, but would have to do it with
    # vectors of size d, which seems like overkill (if this didn't
    # seem like overkill already).
    if f is logsoftmax: return 1.-1./d

    # As is standard, use the gradient of f at zero.  However, since f
    # might not be differentiable at zero (e.g., ReLU), compute
    # gradient a little bit to the left and right and average.

    delta = 0.1
    g = []
    for xv in [-delta, 0., delta]:
        x = constant(0.)
        y = f(x)
        values = {x: xv}
        gradients = {x: 0., y: 1.}
        y.forward(values)
        y.backward(values, gradients)
        g.append(gradients[x])

    if abs(g[2]-g[0])/2. > delta:
        return (g[0] + g[2]) / 2.
    else:
        return g[1]

def make_layer(input_dims, output_dims, f=tanh, bias=True, model=None):
    """Make a fully-connected layer.

    input_dims:  Input size or sequence of input sizes. If an input
                 size is n > 0, then that input will expect an
                 n-dimensional vector. If an input size is n < 0, then
                 that input will expect an integer in [0, n), which
                 you can either think of as a one-hot vector or as an
                 index into a lookup table.
    output_dims: Output size.
    f:           Activation function (default tanh)."""

    if type(input_dims) is int: input_dims = [input_dims]
    if model is None: model = parameter.all

    gain = guess_gain(f, output_dims)
    def random(variance, shape):
        return numpy.random.normal(0., variance, shape)
        #return numpy.random.uniform(-variance*3**0.5, variance*3**0.5, shape)
    
    # Although it is more conventional to left-multiply by the weight
    # matrix, we right-multiply so it works correctly with stacks of
    # vectors.

    weight = []
    for a, d in enumerate(input_dims):
        if d >= 0:
            variance = ((d + output_dims)/2.)**-0.5 / gain
            w = parameter(random(variance, (d, output_dims)), model=model)
            weight.append(w)
        else:
            variance = output_dims**-0.5 / gain
            w = [parameter(random(variance, (output_dims,)), model=model) for i in xrange(-d)]
            weight.append(w)
    if bias:
        b = parameter(numpy.zeros((output_dims,)), model=model)

    def layer(*args):
        if len(args) != len(weight):
            raise TypeError("wrong number of inputs")
        s = b if bias else constant(0.)
        for w, x in zip(weight, args):
            if isinstance(x, int):
                s += w[x]
            else:
                s += dot(x, w)
        if f:
            return f(s)
        else:
            return s
    return layer

