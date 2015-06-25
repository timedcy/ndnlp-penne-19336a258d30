"""Convolution and pooling."""

import numpy, nphacks
from expr import *

class convolve(Expression):
    """Discrete convolution. Same as scipy.signal.convolve (n-dimensional)
    if available; otherwise, same as numpy.convolve (1-dimensional)."""

    def __init__(self, x, y, mode='full'):
        Expression.__init__(self, x, y)
        self.mode = mode

    def forward(self, values):
        values[self] = nphacks.convolve(values[self.args[0]], values[self.args[1]], mode=self.mode)

    def backward(self, values, gradients):
        x, y = values[self.args[0]], values[self.args[1]]
        gz = gradients[self]
        if self.mode != 'full':
            # Since 'valid' and 'same' are equivalent to doing a
            # 'full' convolution followed by clipping, the backward
            # step just needs to first pad the same number of zeros.
            if self.mode == 'valid':
                pad = [n-1 for n in y.shape]
            elif self.mode == 'same':
                pad = [((n-1)/2, n/2) for n in y.shape]
            gz = numpy.pad(gz, pad, mode='constant')

        if self.args[0] in gradients:
            gradients[self.args[0]] += nphacks.correlate(gz, y.conj(), mode='valid')
        if self.args[1] in gradients:
            gradients[self.args[1]] += nphacks.correlate(gz, x.conj(), mode='valid')

class pool(Expression):
    """Tile an array into nonoverlapping blocks and return an array of
    the blocks.
    
    a:          The array to be tiled.
    blockshape: The shape of the blocks.

    If blockshape = (a, b, ...), then
      result[I, J, ..., i, j, ...] = a[I*a+i, J*b+j, ...]
    """

    # To do: overlapping windows

    def __init__(self, a, blockshape):
        Expression.__init__(self, a)
        self.blockshape = blockshape
        rk = len(blockshape)
        self.forward_axes = range(0, rk*2, 2) + range(1, rk*2, 2)
        self.backward_axes = []
        for i in xrange(rk):
            self.backward_axes.extend([i,i+rk])

    def forward(self, values):
        # http://stackoverflow.com/questions/10896841/find-a-3x3-sliding-window-over-an-image
        vx = values[self.args[0]]
        if vx.ndim != len(self.blockshape):
            raise ValueError("block and value must have same rank")
        newshape = []
        for vd, bd in zip(vx.shape, self.blockshape):
            if vd % bd != 0:
                raise ValueError("block size must divide value size")
            newshape.extend([vd/bd, bd])
        values[self] = vx.reshape(newshape).transpose(self.forward_axes)

    def backward(self, values, gradients):
        arg = self.args[0]
        if arg in gradients:
            g = gradients[self]
            g = g.transpose(self.backward_axes)
            g = g.reshape(values[arg].shape)
            gradients[arg] += g

def max_pool(a, blockshape):
    """Tile an array into nonoverlapping blocks and return the maximum
    value of each block."""

    rk = len(blockshape)
    return amax(pool(a, blockshape), axis=tuple(range(rk, rk*2)))

def mean_pool(a, blockshape):
    """Tile an array into nonoverlapping blocks and return the mean
    value of each block."""

    rk = len(blockshape)
    return mean(pool(a, blockshape), axis=tuple(range(rk, rk*2)))

if __name__ == "__main__":
    import compute
    from expr import parameter

    a = parameter(numpy.random.uniform(-1., 1., (16, 16)))
    b = convolve(a, constant(numpy.random.uniform(-1., 1., (3, 3))), mode='full')
    c = max_pool(b, (2, 2))
    #d = b * constant(numpy.random.uniform(-1., 1., (18, 18)))
    o = asum(c)
    
    values = compute.compute_values(o)

    auto = compute.compute_gradients(o, values)[a]
    check = compute.check_gradients(o)[a]

    d = (auto-check).ravel()
    print "error:", d.dot(d)
