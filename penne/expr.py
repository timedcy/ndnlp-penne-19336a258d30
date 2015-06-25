"""Building and using neural networks."""

import logging
import numpy, nphacks
import operator
import traceback # for expression tracebacks
import subprocess # for calling graphviz

__all__ = ["Expression", "Unary", "Binary", "Reduction",
           "constant", "one_hot",
           "parameter", "load_model", "save_model",
           "add", "subtract", "multiply", "divide", "negative", "power",
           "maximum", "minimum", "clip",
           "less", "less_equal", "greater", "greater_equal",
           "logical_and", "logical_or", "logical_not",
           "exp", "log", "tanh",
           "asum", "amax", "amin", "mean",
           "dot", "einsum",
           "concatenate", "stack", "transpose", "reshape",
           "topological", "graphviz"]

class Expression(object):
    """Base class for expression classes."""
    serial = 0
    def __init__(self, *args):
        self.args = args
        if logging.debug:
            self.stack = logging.extract_stack()
        self.serial = Expression.serial
        Expression.serial += 1

    def forward(self, values):
        """Compute value of self given values of args."""
        raise NotImplementedError()

    def backward(self, values, gradients):
        """Update gradient with respect to args given gradient with respect to self.

        pre/post: gradients[self] has the same shape as values[self].
        pre: gradients[self.args[i]] is either 0.0 or has the same shape as values[self.args[i]].
        post: gradients[self.args[i]] has the same shape as values[self.args[i]].
        """
        pass

    def __getitem__(self, item): return getitem(self, item)

    def __add__(self, other):    return add(self, other)
    def __sub__(self, other):    return subtract(self, other)
    def __mul__(self, other):    return multiply(self, other)
    def __div__(self, other):    return divide(self, other)
    def __pow__(self, other):    return power(self, other)
    def __neg__(self):           return negative(self)

    def __lt__(self, other):     return less(self, other)
    def __le__(self, other):     return less_equal(self, other)
    def __gt__(self, other):     return greater(self, other)
    def __ge__(self, other):     return greater_equal(self, other)

    def dot(self, other):        return dot(self, other)

    def __str__(self):
        args = ["<%s>" % arg.serial for arg in self.args]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(args))

    def _repr_png_(self):
        """IPython magic: show PNG representation of the transducer.
        Adapted from pyfst."""
        process = subprocess.Popen(['dot', '-Tpng'], 
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        out, err = process.communicate(graphviz(self))
        if err:
            raise Exception(err)
        return out

class constant(Expression):
    """A constant value."""
    def __init__(self, value):
        Expression.__init__(self)
        self.value = value

    def forward(self, values):
        values[self] = self.value

    def __str__(self):
        return "constant(%s)" % (self.value,)
        
def one_hot(dims, i):
    """A one-hot vector."""
    result = numpy.zeros((dims,))
    result[i] = 1.
    return constant(result)

## Parameters and models

class parameter(Expression):
    """A parameter that is to be trained."""

    # Global list of all parameter objects.
    all = []

    def __init__(self, value, model=all):
        Expression.__init__(self)
        self.value = value
        model.append(self)

    def forward(self, values):
        values[self] = self.value

    def __str__(self):
        return "parameter(%s)" % (self.value,)

def save_model(outfile, model=parameter.all):
    """Saves parameters to a file."""
    for param in model:
        numpy.save(outfile, param.value)

def load_model(infile, model=parameter.all):
    """Loads parameters from a file.

    Assumes that there are exactly the same number of parameters as
    when the model was saved, created in the same order and with the
    same shapes."""

    for param in model:
        param.value[...] = numpy.load(infile)

class Unary(Expression):
    """Base class for unary componentwise operations."""
    def __init__(self, x):
        Expression.__init__(self, x)

    def forward(self, values):
        x = self.args[0]
        values[self] = self.f(values[x])

    def backward(self, values, gradients):
        x = self.args[0]
        dfdx = self.dfdx(values[x], values[self])
        if x in gradients:
            gradients[x] += nphacks.contract_like(dfdx * gradients[self], values[x])

class Binary(Expression):
    """Base class for binary componentwise operations."""
    def __init__(self, x, y):
        Expression.__init__(self, x,  y)

    def forward(self, values):
        x, y = self.args
        values[self] = self.f(values[x], values[y])

    def backward(self, values, gradients):
        x, y = self.args
        dfdx = self.dfdx(values[x], values[y], values[self])
        dfdy = self.dfdy(values[x], values[y], values[self])
        if x in gradients:
            gradients[x] += nphacks.contract_like(dfdx * gradients[self], values[x])
        if y in gradients:
            gradients[y] += nphacks.contract_like(dfdy * gradients[self], values[y])

## Arithmetic operations

class add(Binary):
    f = staticmethod(numpy.add)

    # "Inline" df to avoid multiplications by 1 in this very common case
    def backward(self, values, gradients):
        x, y = self.args
        if x in gradients:
            gradients[x] += nphacks.contract_like(gradients[self], values[x])
        if y in gradients:
            gradients[y] += nphacks.contract_like(gradients[self], values[y])

class subtract(Binary):
    f = staticmethod(numpy.subtract)
    dfdx = staticmethod(lambda x,y,z: 1.)
    dfdy = staticmethod(lambda x,y,z: -1.)

class negative(Unary):
    f = staticmethod(numpy.negative)
    dfdx = staticmethod(lambda x,z: -1.)

class multiply(Binary):
    f = staticmethod(numpy.multiply)
    dfdx = staticmethod(lambda x,y,z: y)
    dfdy = staticmethod(lambda x,y,z: x)

class divide(Binary):
    f = staticmethod(numpy.divide)
    dfdx = staticmethod(lambda x,y,z: 1./y)
    dfdy = staticmethod(lambda x,y,z: -x/y**2)

class power(Binary):
    f = staticmethod(numpy.power)
    dfdx = staticmethod(lambda x,y,z: z*y/x)
    dfdy = staticmethod(lambda x,y,z: z*numpy.log(x))

class log(Unary):
    f = staticmethod(numpy.log)
    dfdx = staticmethod(lambda x,z: 1./x)

class exp(Unary):
    f = staticmethod(numpy.exp)
    dfdx = staticmethod(lambda x,z: z)

class tanh(Unary):
    f = staticmethod(numpy.tanh)
    dfdx = staticmethod(lambda x,z: 1.-z**2)

class maximum(Binary):
    f = staticmethod(numpy.maximum)
    dfdx = staticmethod(lambda x,y,z: (x > y).astype(float))
    dfdy = staticmethod(lambda x,y,z: (x <= y).astype(float))

class minimum(Binary):
    f = staticmethod(numpy.minimum)
    dfdx = staticmethod(lambda x,y,z: (x < y).astype(float))
    dfdy = staticmethod(lambda x,y,z: (x >= y).astype(float))

def clip(x, lo, hi):
    return minimum(maximum(x, lo), hi)

## Conditionals

class where(Expression):
    def forward(self, values):
        c, x, y = self.args
        values[self] = numpy.where(values[c], values[x], values[y])

    def backward(self, values, gradients):
        c, x, y = self.args
        if x in gradients:
            gradients[x] += nphacks.contract_like(c.astype(bool) * gradients[self], values[x])
        if y in gradients:
            gradients[y] += nphacks.contract_like(numpy.logical_not(c) * gradients[self], values[x])

class greater(Binary):
    f = staticmethod(numpy.greater)
class greater_equal(Binary):
    f = staticmethod(numpy.greater_equal)
class less(Binary):
    f = staticmethod(numpy.less)
class less_equal(Binary):
    f = staticmethod(numpy.less_equal)
class logical_and(Binary):
    f = staticmethod(numpy.logical_and)
class logical_or(Binary):
    f = staticmethod(numpy.logical_or)
class logical_not(Binary):
    f = staticmethod(numpy.logical_not)

### Reductions

class Reduction(Expression):
    """Base class for reduction operations."""

    def __init__(self, x, axis=None, keepdims=False):
        Expression.__init__(self, x)
        self.axis = axis
        self.keepdims = keepdims
        if not keepdims:
            self.raxis = nphacks.restore_axis(axis)

    def forward(self, values):
        values[self] = self.f(values[self.args[0]], axis=self.axis, keepdims=self.keepdims)

    def backward(self, values, gradients):
        arg = self.args[0]
        if arg in gradients:
            if self.keepdims:
                gself = gradients[self]
                vself = values[self]
            else:
                gself = numpy.asarray(gradients[self])[self.raxis]
                vself = numpy.asarray(values[self])[self.raxis]
            d = self.df(values[arg], vself, axis=self.axis)
            gradients[arg] += nphacks.expand_like(gself * d, values[arg])

class asum(Reduction):
    """Sum elements of an array, same as numpy.sum."""
    f = staticmethod(numpy.sum)
    df = staticmethod(lambda x,y,axis: 1.)

class amax(Reduction):
    f = staticmethod(numpy.amax)
    @staticmethod
    def df(x, y, axis):
        is_max = x == y
        return is_max / numpy.sum(is_max, axis=axis, keepdims=True)

class amin(Reduction):
    f = staticmethod(numpy.amin)
    @staticmethod
    def df(x, y, axis):
        is_min = x == y
        return is_min / numpy.sum(is_min, axis=axis, keepdims=True)

class mean(Reduction):
    f = staticmethod(numpy.mean)
    df = staticmethod(lambda x,y,axis: float(fx.size) / x.size)

### Product-like operations

class dot(Expression):
    def __init__(self, x, y):
        Expression.__init__(self, x, y)

    def forward(self, values):
        xv = values[self.args[0]]
        yv = values[self.args[1]]
        values[self] = numpy.dot(xv, yv)

    def backward(self, values, gradients):
        x, y = self.args
        xd, yd = values[x].ndim, values[y].ndim

        # Common cases

        if xd <= 1 and yd <= 1:
            if x in gradients:
                gradients[x] += numpy.dot(gradients[self], values[y])
            if y in gradients:
                gradients[y] += numpy.dot(values[x], gradients[self])

        elif xd == 2 and yd == 1:
            if x in gradients:
                gradients[x] = nphacks.add_outer(gradients[x], gradients[self], values[y])
            if y in gradients:
                gradients[y] += numpy.dot(values[x].T, gradients[self])

        elif xd == 1 and yd == 2:
            if x in gradients:
                gradients[x] += numpy.dot(gradients[self], values[y].T)
            if y in gradients:
                gradients[y] = nphacks.add_outer(gradients[y], values[x], gradients[self])

        elif xd == 2 and yd == 2:
            if x in gradients:
                gradients[x] = nphacks.add_dot(gradients[x], gradients[self], values[y].T)
            if y in gradients:
                gradients[y] = nphacks.add_dot(gradients[y], values[x].T, gradients[self])

        else:
            xv, yv = values[x], values[y].swapaxes(-2, 0)
            xr = xv.reshape((-1, xv.shape[-1]))
            yr = yv.reshape((yv.shape[0], -1))
            gzr = gradients[self].reshape((xr.shape[0], yr.shape[1]))
            if x in gradients:
                gradients[x] += numpy.dot(gzr, yr.T).reshape(xv.shape)
            if y in gradients:
                gradients[y] += numpy.dot(xr.T, gzr).reshape(yv.shape).swapaxes(-2, 0)

class einsum(Expression):
    def __init__(self, subscripts, *args):
        Expression.__init__(self, *args)
        self.subscripts = subscripts
        lhs, rhs = subscripts.split("->")
        self.bsubscripts = []
        for i in lhs.split(","):
            self.bsubscripts.append("%s,%s->%s" % (lhs, rhs, i))

    def forward(self, values):
        values[self] = numpy.einsum(self.subscripts, *[values[arg] for arg in self.args])
    def backward(self, values, gradients):
        args = [values[arg] for arg in self.args] + [gradients[self]]
        for i in xrange(len(self.args)):
            if self.args[i] in gradients:
                args[i] = nphacks.broadcast_to(1., values[self.args[i]].shape)
                gradients[self.args[i]] += numpy.einsum(self.bsubscripts[i], *args)
                args[i] = values[self.args[i]]

### Cutting and pasting

class concatenate(Expression):
    """Concatenate arrays along an axis, same as numpy.concatenate."""
    def __init__(self, args, axis=0):
        Expression.__init__(self, *args)
        self.axis = axis
    def forward(self, values):
        values[self] = numpy.concatenate([values[arg] for arg in self.args], axis=self.axis)
    def backward(self, values, gradients):
        i = 0
        s = [slice(None)]*gradients[self].ndim
        for arg in self.args:
            d = values[arg].shape[self.axis]
            if arg in gradients:
                s[self.axis] = slice(i, i+d)
                gradients[arg] += gradients[self][s]
            i += d

class stack(Expression):
    """Stack arrays along a new axis, same as numpy.stack."""
    def __init__(self, args, axis=0):
        Expression.__init__(self, *args)
        self.axis = axis
    def forward(self, values):
        values[self] = nphacks.stack([values[arg] for arg in self.args], axis=self.axis)
    def backward(self, values, gradients):
        s = [slice(None)]*gradients[self].ndim
        for i, arg in enumerate(self.args):
            if arg in gradients:
                s[self.axis] = i
                gradients[arg] += gradients[self][s]

class reshape(Expression):
    """Reshape an array, same as numpy.reshape."""
    def __init__(self, arg, shape):
        Expression.__init__(self, arg)
        self.shape = shape
    def forward(self, values):
        values[self] = numpy.reshape(values[self.args[0]], self.shape)
    def backward(self, values, gradients):
        arg = self.args[0]
        if arg in gradients:
            gradients[arg] += numpy.reshape(gradients[self], values[arg].shape)

class transpose(Expression):
    """Transpose all axes, same as numpy.transpose."""
    def __init__(self, arg, axes=None):
        Expression.__init__(self, arg)
        self.axes = axes
        if axes:
            self.raxes = [i for (i,j) in sorted(enumerate(axes), key=operator.itemgetter(1))]
        else:
            self.raxes = None
    def forward(self, values):
        values[self] = numpy.transpose(values[self.args[0]], self.axes)
    def backward(self, values, gradients):
        arg = self.args[0]
        if arg in gradients:
            gradients[arg] += numpy.transpose(gradients[self], self.raxes)

class getitem(Expression):
    def __init__(self, arg, item):
        """item can be an integer index or a slice (not an Expression)."""
        Expression.__init__(self, arg)
        if isinstance(item, Expression): raise TypeError("item should be an index or slice")
        self.item = item

    def forward(self, values):
        arg = values[self.args[0]]
        values[self] = arg[self.item]
    def backward(self, values, gradients):
        arg = self.args[0]
        if arg in gradients:
            if not isinstance(gradients[arg], numpy.ndarray):
                gradients[arg] = numpy.zeros_like(values[arg])
            gradients[arg][self.item] += gradients[self]

    def __str__(self):
        item = self.item
        if isinstance(item, slice): 
            start = item.start or ""
            stop = item.stop or ""
            if item.stride is not None:
                item = "%s:%s:%s" % (start, stop, item.stride)
            else:
                item = "%s:%s" % (start, stop)
        return "%s[%s]" % (self.args[0], item)

###

def topological(x):
    """Traverse an Expression in topological (bottom-up) order."""

    visited = set()
    result = []
    def visit(subx):
        if subx in visited: return
        visited.add(subx)
        for arg in subx.args:
            visit(arg)
        result.append(subx)
    visit(x)
    return result

### Visualization

def graphviz(x):
    """Draw the computation graph of an expression in the DOT language.

    x: the expression to draw
    outfile: file to write to

    The output can be processed using GraphViz's dot command.
    """
    result = []
    result.append("digraph {\n")
    result.append('  rankdir=BT;\n')
    result.append('  node [shape=box,margin=0.1,width=0,height=0,style=filled,fillcolor=lightgrey,penwidth=0,fontname="Courier",fontsize=10];\n')
    result.append("  edge [arrowsize=0.5];\n")
    for subx in topological(x):
        result.append('  %s [label="%s: %s"];\n' % (subx.serial, subx.serial, subx.__class__.__name__))
        for arg in subx.args:
            result.append("  %s -> %s\n" % (arg.serial, subx.serial))
    result.append("}\n")
    return "".join(result)
