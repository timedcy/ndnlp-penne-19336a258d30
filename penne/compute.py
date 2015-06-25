"""Compute values, gradients, and other things."""

__all__ = ['compute_values', 'compute_gradients', 'check_gradients']

import expr # parameter, topological
import logging

import numpy, nphacks
import sys
import collections

### Forward and backward propagation

def format_value(v, indent=4):
    result = []
    newline = '\n' + ' '*indent
    s = str(v).split('\n')
    if len(s) > 1: result.append(newline)
    result.append(newline.join([]+s))
    result.append('\n')
    return ''.join(result)
    
def compute_values(x, initvalues={}):
    """Evaluate an expression and all its subexpressions.

    x:          The expression to evaluate.
    initvalues: Optional dictionary from subexpressions to
                precomputed values; can be used to continue a
                computation when the expression grows.
    """
    values = collections.OrderedDict()

    for subx in expr.topological(x):
        if subx in initvalues:
            values[subx] = initvalues[subx]
        else:
            try:
                subx.forward(values)
            except:
                if logging.debug:
                    sys.stderr.write("Expression traceback (most recent call last):\n" + "".join(logging.format_list(subx.stack)))
                raise
            if logging.trace:
                sys.stdout.write("<%s> = %s = %s" % (subx.serial, subx, format_value(values[subx])))

    return values

def compute_gradients(x, values=None):
    """Compute gradients using automatic differentiation.

    x:      The expression to compute gradients of.
    values: As returned by compute_values(x); contains all values to compute
            gradients with respect to.
    """

    if values is None:
        values = compute_values(x)
    if not isinstance(values, collections.OrderedDict): raise TypeError()
    if x not in values: raise ValueError()

    gradients = {x: 1.}
    for subx in values:
        if isinstance(subx, expr.parameter) or any(arg in gradients for arg in subx.args):
            if subx not in gradients:
                gradients[subx] = 0.

    for subx in reversed(values): # assume values is in topological order
        if subx not in gradients:
            continue
        if logging.trace:
            sys.stdout.write("d<%s>/d<%s> = %s" % (x.serial, subx.serial, format_value(gradients[subx])))
        try:
            subx.backward(values, gradients)
            if logging.trace:
                for arg in subx.args:
                    sys.stdout.write("    d<%s>/d<%s> := %s" % (x.serial, arg.serial, format_value(gradients[arg], indent=8)))
        except:
            if logging.debug:
                sys.stderr.write("Expression traceback (most recent call last):\n" + "".join(logging.format_list(subx.stack)))
            raise
    return gradients

def check_gradients(x, model=None, delta=1e-10):
    """Compute gradients using symmetric differences.

    x:     The expression to compute gradients of.
    model: Optional list of parameters to compute gradients with respect to.

    This is extremely slow and is used only for debugging."""

    if model is None:
        model = [subx for subx in expr.topological(x) if isinstance(subx, expr.parameter)]

    gradients = {}
    for param in model:
        g = numpy.empty_like(param.value)
        it = numpy.nditer([param.value, g], [], [['readwrite'], ['writeonly']])
        for pi, gi in it:
            save = pi
            pi -= delta/2
            val_minus = compute_values(x)[x]
            pi += delta
            val_plus = compute_values(x)[x]
            pi[...] = save
            gi[...] = (val_plus-val_minus)/delta
        gradients[param] = g
    return gradients

