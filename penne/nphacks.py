"""Various operations on NumPy arrays."""

import numpy
try:
    import scipy.linalg.blas
    use_blas = True
except ImportError:
    use_blas = False

try:
    # N-dimensional
    from scipy.signal import convolve, correlate
except ImportError:
    # 1-dimensional
    def convolve(a, v, mode='full'):
        # bypass numpy.convolve because we don't want it to swap arguments
        return numpy.correlate(a, v[::-1].conj(), mode)
    correlate = numpy.correlate

def contract_like(a, b):
    """Sum axes of a so that shape is the same as b.

    If y = broadcast_to(x, yshape), gx = contract_like(gy, x).
    """

    if numpy.shape(a) == numpy.shape(b):
        return a
    b1 = broadcast_to(b, a.shape)
    axis = [i for i in xrange(b1.ndim) if b1.strides[i] == 0] # what if b already had any zero strides?
    return numpy.sum(a, axis=tuple(axis)).reshape(numpy.shape(b))

def expand_like(a, b):
    return numpy.array(broadcast_to(a, numpy.shape(b)))

def restore_axis(axis):
    if axis is None:
        return Ellipsis
    elif isinstance(axis, int):
        return [slice(None)] * (axis)-1 + [numpy.newaxis]
    elif isinstance(axis, tuple):
        raxis = [slice(None)] * (max(axis)+1)
        for i in axis:
            raxis[i] = numpy.newaxis
        return raxis

def add_outer(a, x, y):
    """Add the outer product of x and y to a, possibly overwriting a.
    a = add_outer(a, x, y) is equivalent to a += numpy.outer(x, y)."""

    if use_blas and isinstance(a, numpy.ndarray):
        if numpy.isfortran(a.T):
            scipy.linalg.blas.dger(1., y, x, a=a.T, overwrite_a=1)
            return a
        elif numpy.isfortran(a):
            scipy.linalg.blas.dger(1., x, y, a=a, overwrite_a=1)
            return a

    # einsum is written in C and is faster than outer
    a += numpy.einsum('i,j->ij', x, y)
    return a

def add_dot(c, a, b):
    if use_blas and isinstance(c, numpy.ndarray):
        if numpy.isfortran(c.T):
            scipy.linalg.blas.dgemm(1., b.T, a.T, 1., c.T, overwrite_c=True)
            return c
        elif numpy.isfortran(c):
            scipy.linalg.blas.dgemm(1., a, b, 1., c, overwrite_c=True)
            return c

    c += numpy.dot(a, b)
    return c

# Will be added in NumPy 1.10; this is copied from there

if 'stack' in dir(numpy):
    stack = numpy.stack
else:
    def stack(arrays, axis=0):
        arrays = [numpy.asarray(arr) for arr in arrays]
        if not arrays:
            raise ValueError('need at least one array to stack')

        shapes = set(arr.shape for arr in arrays)
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')

        result_ndim = arrays[0].ndim + 1
        if not -result_ndim <= axis < result_ndim:
            msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
            raise IndexError(msg)
        if axis < 0:
            axis += result_ndim

        sl = (slice(None),) * axis + (numpy.newaxis,)
        expanded_arrays = [arr[sl] for arr in arrays]
        return numpy.concatenate(expanded_arrays, axis=axis)

# Will be defined in NumPy 1.10
if 'broadcast_to' in dir(numpy):
    broadcast_to = numpy.broadcast_to
else:
    def broadcast_to(array, shape):
        shape = tuple(shape) if numpy.iterable(shape) else (shape,)
        if not shape and array.shape:
            raise ValueError('cannot broadcast a non-scalar to a scalar array')
        if any(size < 0 for size in shape):
            raise ValueError('all elements of broadcast shape must be non-'
                             'negative')
        broadcast = numpy.nditer(
            (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'],
            op_flags=['readonly'], itershape=shape, order='C').itviews[0]
        return broadcast
