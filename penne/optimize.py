"""Training neural networks."""

__all__ = ['StochasticGradientDescent', 'SGD', 'Adagrad', 'Adadelta', 'Momentum', 'Nesterov']

import numpy
import compute
import expr # parameter

class StochasticGradientDescent(object):
    """Stochastic gradient descent."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def receive(self, x):
        values = compute.compute_values(x)
        gradients = compute.compute_gradients(x, values)
        for param in gradients:
            if isinstance(param, expr.parameter):
                param.value -= self.learning_rate * gradients[param]
        return values[x]
    
SGD = StochasticGradientDescent

class Adagrad(object):
    """Adagrad (diagonal version).

    John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient
    methods for online learning and stochastic optimization. JMLR
    12:2121-2159, 2011.

    delta: Starting value for sum of squared gradients. Theoretically,
           it can be zero, but in practice, it must be positive.
    """

    def __init__(self, learning_rate=1., delta=1.):
        self.learning_rate = learning_rate
        self.sum_gradients2 = {}
        self.delta = delta

    def receive(self, x):
        values = compute.compute_values(x)
        gradients = compute.compute_gradients(x, values)
        for param in gradients:
            if isinstance(param, expr.parameter):
                if param not in self.sum_gradients2:
                    self.sum_gradients2[param] = numpy.full_like(param.value, self.delta)
                self.sum_gradients2[param] += gradients[param] ** 2
                param.value -= self.learning_rate * gradients[param] / numpy.sqrt(self.sum_gradients2[param])
        return values[x]

class Adadelta(object):
    """Adadelta.

    Matthew D. Zeiler. ADADELTA: An adaptive learning rate
    method. arXiv:1212.5701, 2012.
    """

    def __init__(self, decay=0.95, epsilon=1e-6):
        self.ave_gradients2 = {}
        self.ave_updates2 = {}
        self.decay = decay
        self.epsilon = epsilon

    def receive(self, x):
        values = compute.compute_values(x)
        gradients = compute.compute_gradients(x, values)
        for param in gradients:
            if isinstance(param, expr.parameter):

                if param not in self.ave_gradients2:
                    self.ave_gradients2[param] = numpy.zeros_like(param.value)
                if param not in self.ave_updates2:
                    self.ave_updates2[param] = numpy.zeros_like(param.value)

                self.ave_gradients2[param] *= self.decay
                self.ave_gradients2[param] += (1-self.decay) * gradients[param] ** 2

                update = gradients[param] * numpy.sqrt((self.ave_updates2[param] + self.epsilon) / 
                                                       (self.ave_gradients2[param] + self.epsilon))

                self.ave_updates2[param] *= self.decay
                self.ave_updates2[param] += (1-self.decay) * update ** 2

                param.value -= update

        return values[x]

class Momentum(object):
    """Stochastic gradient descent with momentum."""

    def __init__(self, learning_rate=0.01, momentum_coeff=0.9):
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff

        self.velocities = {}

    def receive(self, x):
        values = compute.compute_values(x)
        gradients = compute.compute_gradients(x, values)
        for param in self.velocities:
            self.velocities[param] *= self.momentum_coeff
        for param in gradients:
            if isinstance(param, expr.parameter):
                if param not in self.velocities:
                    self.velocities[param] = numpy.zeros_like(param.value)
                self.velocities[param] -= self.learning_rate * gradients[param]
        for param in self.velocities:
            param.value += self.velocities[param]
        return values[x]

class Nesterov(object):
    """Momentum-like version of Nesterov accelerated gradient.
    
    Ilya Sutskever, James Martens, George Dahl, and Geoffrey
    Hinton. On the importance of initialization and momentum in deep
    learning. In Proc. ICML, 2013."""

    def __init__(self, learning_rate=0.01, momentum_coeff=0.9):
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.velocities = {}

    def receive(self, x):
        for param in self.velocities:
            self.velocities[param] *= self.momentum_coeff
            param.value += self.velocities[param]
        values = compute.compute_values(x)
        gradients = compute.compute_gradients(x, values)
        for param in gradients:
            if isinstance(param, expr.parameter):
                g = self.learning_rate * gradients[param]
                if param not in self.velocities:
                    self.velocities[param] = numpy.zeros_like(param.value)
                self.velocities[param] -= g
                param.value -= g
        return values[x]
