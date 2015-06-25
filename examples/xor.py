import sys
sys.path.append("..")
from penne import *
import numpy

num_hidden = 3

a1 = make_layer(2, num_hidden)
a2 = make_layer(num_hidden, 1)
def xor(x, y):
    i = numpy.array([x,y])
    h = a1(constant(i))
    return a2(h)

trainer = SGD(learning_rate=0.1)

data = []
for x in [-1, +1]:
    for y in [-1, +1]:
        data.append((x,y,+1 if x != y else -1))

#load_model(open("xor.npy"))
first = True

for epoch in xrange(100):
    epoch_loss = 0.
    for x, y, z in data:
        correct = numpy.array([z])
        guess = xor(x, y)
        loss = distance2(guess, constant(correct))

        # Generate visualization
        if first:
            first = False
            with open("xor.dot", "w") as outfile:
                outfile.write(graphviz(loss))
        
        # Check gradient
        values = compute_values(loss)
        auto = compute_gradients(loss, values)
        check = check_gradients(loss)
        for p in check:
            print "auto:", auto[p]
            print "check:", check[p]

        epoch_loss += trainer.receive(loss)
    print "epoch=%s loss=%s" % (epoch, epoch_loss/len(data))

save_model(open("xor.npy", "w"))
